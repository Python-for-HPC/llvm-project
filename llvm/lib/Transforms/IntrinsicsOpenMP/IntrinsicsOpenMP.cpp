//===- IntrinsicsOpenMP.cpp - Codegen OpenMP from IR intrinsics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements code generation for OpenMP from intrinsics embedded in
// the IR, using the OpenMPIRBuilder
//
//===-------------------------------------------------------------------------===//

#include "llvm-c/Transforms/IntrinsicsOpenMP.h"
#include "CGIntrinsicsOpenMP.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IntrinsicsOpenMP/IntrinsicsOpenMP.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace omp;
using namespace iomp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

struct IntrinsicsOpenMP : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  IntrinsicsOpenMP() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // NOTE: Codegen for nested or combined constructs assumes code is generated
    // bottom-up, that is from the innermost directive to the outermost. This
    // simplifies handling of DSA attributes by avoiding renaming values when an
    // outlined function privatizes them in the DSAValueMap. For combined
    // constructs, codegen enforces the innermost-to-outermost order. For nested
    // constructs, the assumption is that LLVM IR traversal for region tags
    // follows that order.
    LLVM_DEBUG(dbgs() << "=== Start IntrinsicsOpenMPPass v4\n");

    Function *RegionEntryF = M.getFunction("llvm.directive.region.entry");

    // Return early for lack of directive intrinsics.
    if (!RegionEntryF) {
      LLVM_DEBUG(dbgs() << "No intrinsics directives, exiting...\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "=== Dump Module\n"
                      << M << "=== End of Dump Module\n");

    CGIntrinsicsOpenMP CGIOMP(M);
    // Iterate over all calls to directive intrinsics and transform code
    // using OpenMPIRBuilder for lowering.
    SmallVector<User *, 4> RegionEntryUsers(RegionEntryF->users());
    for (User *Usr : RegionEntryUsers) {
      LLVM_DEBUG(dbgs() << "Found Usr " << *Usr << "\n");
      CallBase *CBEntry = dyn_cast<CallBase>(Usr);
      assert(CBEntry && "Expected call to region entry intrinsic");

      // Extract the directive kind and data sharing attributes of values
      // from the operand bundles of the intrinsic call.
      Directive Dir = OMPD_unknown;
      SmallVector<OperandBundleDef, 16> OpBundles;
      DSAValueMapTy DSAValueMap;

      OMPLoopInfoStruct OMPLoopInfo;
      ParRegionInfoStruct ParRegionInfo;
      TargetInfoStruct TargetInfo;
      TeamsInfoStruct TeamsInfo;

      MapVector<Value *, SmallVector<FieldMappingInfo, 4>> StructMappingInfoMap;

      bool IsDeviceTargetRegion = false;

      CBEntry->getOperandBundlesAsDefs(OpBundles);
      // TODO: parse clauses.
      for (OperandBundleDef &O : OpBundles) {
        StringRef Tag = O.getTag();
        LLVM_DEBUG(dbgs() << "OPB " << Tag << "\n");

        // TODO: check for conflicting DSA, for example reduction variables
        // cannot be set private. Should be done in Numba.
        if (Tag.startswith("DIR")) {
          auto It = StringToDir.find(Tag);
          assert(It != StringToDir.end() && "Directive is not supported!");
          Dir = It->second;
        } else if (Tag.startswith("QUAL")) {
          const ArrayRef<Value *> &TagInputs = O.inputs();
          if (Tag.startswith("QUAL.OMP.NORMALIZED.IV")) {
            assert(O.input_size() == 1 && "Expected single IV value");
            OMPLoopInfo.IV = TagInputs[0];
          } else if (Tag.startswith("QUAL.OMP.NORMALIZED.START")) {
            assert(O.input_size() == 1 && "Expected single START value");
            OMPLoopInfo.Start = TagInputs[0];
          } else if (Tag.startswith("QUAL.OMP.NORMALIZED.LB")) {
            assert(O.input_size() == 1 && "Expected single LB value");
            OMPLoopInfo.LB = TagInputs[0];
          } else if (Tag.startswith("QUAL.OMP.NORMALIZED.UB")) {
            assert(O.input_size() == 1 && "Expected single UB value");
            OMPLoopInfo.UB = TagInputs[0];
          } else if (Tag.startswith("QUAL.OMP.NUM_THREADS")) {
            assert(O.input_size() == 1 && "Expected single NumThreads value");
            ParRegionInfo.NumThreads = TagInputs[0];
          } else if (Tag.startswith("QUAL.OMP.SCHEDULE")) {
            // TODO: Add DIST_SCHEDULE for distribute loops.
            assert(O.input_size() == 1 &&
                   "Expected single chunking scheduling value");
            Constant *Zero = ConstantInt::get(TagInputs[0]->getType(), 0);
            OMPLoopInfo.Chunk = TagInputs[0];

            if (Tag == "QUAL.OMP.SCHEDULE.STATIC") {
              assert(TagInputs[0] == Zero &&
                     "Chunking is not yet supported, requires "
                     "the use of omp_stride (static_chunked)");
              if (TagInputs[0] == Zero)
                OMPLoopInfo.Sched = OMPScheduleType::Static;
              else
                OMPLoopInfo.Sched = OMPScheduleType::StaticChunked;
            } else
              assert(false && "Unsupported scheduling type");
          } else if (Tag.startswith("QUAL.OMP.IF")) {
            assert(O.input_size() == 1 && "Expected single if condition value");
            ParRegionInfo.IfCondition = TagInputs[0];
          } else if (Tag.startswith("QUAL.OMP.REDUCTION.ADD")) {
            DSAValueMap[TagInputs[0]] = DSATypeInfo(DSA_REDUCTION_ADD);
          } else if (Tag.startswith("QUAL.OMP.TARGET.DEV_FUNC")) {
            assert(O.input_size() == 1 &&
                   "Expected a single device function name");
            ConstantDataArray *DevFuncArray =
                dyn_cast<ConstantDataArray>(TagInputs[0]);
            assert(DevFuncArray &&
                   "Expected constant string for the device function");
            TargetInfo.DevFuncName = DevFuncArray->getAsString();
          } else if (Tag.startswith("QUAL.OMP.TARGET.ELF")) {
            assert(O.input_size() == 1 && "Expected a single elf image string");
            ConstantDataArray *ELF = dyn_cast<ConstantDataArray>(TagInputs[0]);
            assert(ELF && "Expected constant string for ELF");
            TargetInfo.ELF = ELF;
          } else if (Tag.startswith("QUAL.OMP.DEVICE")) {
            // TODO: Handle device selection for target regions.
          } else if (Tag.startswith("QUAL.OMP.NUM_TEAMS")) {
            assert(O.input_size() == 1 && "Expected single NumTeams value");
            switch(Dir) {
              case OMPD_target:
                TargetInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_teams:
                TeamsInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
                TargetInfo.NumTeams = TagInputs[0];
                TeamsInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.NumTeams = TagInputs[0];
                TeamsInfo.NumTeams = TagInputs[0];
                break;
              default:
                //report_fatal_error("Unsupported qualifier in directive");
                assert(false && "Unsupported qualifier in directive");
            }
          } else if (Tag.startswith("QUAL.OMP.THREAD_LIMIT")) {
            assert(O.input_size() == 1 && "Expected single ThreadLimit value");
             switch(Dir) {
              case OMPD_target:
                TargetInfo.ThreadLimit = TagInputs[0];
                break;
              case OMPD_teams:
                TeamsInfo.ThreadLimit = TagInputs[0];
                break;
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.ThreadLimit = TagInputs[0];
                TeamsInfo.ThreadLimit = TagInputs[0];
                break;
              default:
                //report_fatal_error("Unsupported qualifier in directive");
                assert(false && "Unsupported qualifier in directive");
            }
          } else /* DSA Qualifiers */ {
            auto It = StringToDSA.find(Tag);
            assert(It != StringToDSA.end() && "DSA type not found in map");
            if (It->second == DSA_MAP_ALLOC_STRUCT ||
                It->second == DSA_MAP_TO_STRUCT ||
                It->second == DSA_MAP_FROM_STRUCT ||
                It->second == DSA_MAP_TOFROM_STRUCT) {
                assert((TagInputs.size() - 1) == 3 &&
                       "Expected input triple for struct mapping");
                Value *Index = TagInputs[1];
                Value *Offset = TagInputs[2];
                Value *NumElements = TagInputs[3];
                StructMappingInfoMap[TagInputs[0]].push_back(
                    {Index, Offset, NumElements, It->second});

                DSAValueMap[TagInputs[0]] = DSATypeInfo(DSA_MAP_STRUCT);
            } else {
                // This firstprivate includes a copy-constructor operand.
                if ((It->second == DSA_FIRSTPRIVATE ||
                     It->second == DSA_LASTPRIVATE) &&
                    TagInputs.size() == 2) {
                Value *V = TagInputs[0];
                ConstantDataArray *CopyFnNameArray =
                    dyn_cast<ConstantDataArray>(TagInputs[1]);
                assert(CopyFnNameArray && "Expected constant string for the "
                                          "copy-constructor function");
                StringRef CopyFnName = CopyFnNameArray->getAsString();
                FunctionCallee CopyConstructor = M.getOrInsertFunction(
                    CopyFnName, V->getType()->getPointerElementType(),
                    V->getType()->getPointerElementType());
                DSAValueMap[TagInputs[0]] =
                    DSATypeInfo(It->second, CopyConstructor);
                } else
                DSAValueMap[TagInputs[0]] = DSATypeInfo(It->second);
            }
          }
        } else if (Tag == "OMP.DEVICE")
          IsDeviceTargetRegion = true;
        else
          report_fatal_error("Unknown tag " + Tag);
      }

      assert(Dir != OMPD_unknown && "Expected valid OMP directive");

      assert(CBEntry->getNumUses() == 1 &&
             "Expected single use of the directive entry CB");
      Use &U = *CBEntry->use_begin();
      CallBase *CBExit = dyn_cast<CallBase>(U.getUser());
      assert(CBExit && "Expected call to region exit intrinsic");
      LLVM_DEBUG(dbgs() << "Found Use of " << *CBEntry << "\n-> AT ->\n"
                        << *CBExit << "\n");

      // Gather info.
      BasicBlock *BBEntry = CBEntry->getParent();
      Function *Fn = BBEntry->getParent();
      const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();

      // Create the basic block structure to isolate the outlined region.
      BasicBlock *StartBB = SplitBlock(BBEntry, CBEntry);
      assert(BBEntry->getUniqueSuccessor() == StartBB &&
             "Expected unique successor at region start BB");

      BasicBlock *BBExit = CBExit->getParent();
      BasicBlock *EndBB = SplitBlock(BBExit, CBExit->getNextNode());
      assert(BBExit->getUniqueSuccessor() == EndBB &&
             "Expected unique successor at region end BB");
      BasicBlock *AfterBB = SplitBlock(EndBB, &*EndBB->getFirstInsertionPt());

      // Define the default BodyGenCB lambda.
      auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                           BasicBlock &ContinuationIP) {
        BasicBlock *CGStartBB = CodeGenIP.getBlock();
        BasicBlock *CGEndBB = SplitBlock(CGStartBB, &*CodeGenIP.getPoint());
        assert(StartBB != nullptr && "StartBB should not be null");
        CGStartBB->getTerminator()->setSuccessor(0, StartBB);
        assert(EndBB != nullptr && "EndBB should not be null");
        EndBB->getTerminator()->setSuccessor(0, CGEndBB);
      };

      // Define the default FiniCB lambda.
      auto FiniCB = [&](InsertPointTy CodeGenIP) {};

      // Remove intrinsics of OpenMP tags, first CBExit to also remove use
      // of CBEntry, then CBEntry.
      CBExit->eraseFromParent();
      CBEntry->eraseFromParent();

      if (Dir == OMPD_parallel) {
        CGIOMP.emitOMPParallel(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                               EndBB, AfterBB, FiniCB, ParRegionInfo);
      } else if (Dir == OMPD_single) {
        CGIOMP.emitOMPSingle(Fn, BBEntry, AfterBB, BodyGenCB, FiniCB);
      } else if (Dir == OMPD_critical) {
        CGIOMP.emitOMPCritical(Fn, BBEntry, AfterBB, BodyGenCB, FiniCB);
      } else if (Dir == OMPD_barrier) {
        CGIOMP.emitOMPBarrier(Fn, BBEntry, OMPD_barrier);
      } else if (Dir == OMPD_for) {
        CGIOMP.emitOMPFor(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                          /* IsStandalone */ true);
        LLVM_DEBUG(dbgs() << "=== For Fn\n" << *Fn << "=== End of For Fn\n");
      } else if (Dir == OMPD_parallel_for) {
        CGIOMP.emitOMPFor(DSAValueMap, OMPLoopInfo, StartBB, BBExit,
                          /* IsStandalone */ false);
        CGIOMP.emitOMPParallel(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                               EndBB, AfterBB, FiniCB, ParRegionInfo);
      } else if (Dir == OMPD_task) {
        CGIOMP.emitOMPTask(DSAValueMap, Fn, BBEntry, StartBB, EndBB, AfterBB);
      } else if (Dir == OMPD_taskwait) {
        CGIOMP.emitOMPTaskwait(BBEntry);
      } else if (Dir == OMPD_target) {
        TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
        CGIOMP.emitOMPTarget(Fn, BBEntry, StartBB, EndBB, DSAValueMap,
                             StructMappingInfoMap, TargetInfo,
                             /* OMPLoopInfo */ nullptr, IsDeviceTargetRegion);
      } else if (Dir == OMPD_teams) {
        CGIOMP.emitOMPTeams(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                            EndBB, AfterBB, TeamsInfo);
      } else if (Dir == OMPD_distribute) {
        CGIOMP.emitOMPDistribute(DSAValueMap, StartBB, BBExit, OMPLoopInfo,
                                 /* IsStandalone */ true);
      } else if (Dir == OMPD_target_teams) {
        TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
        CGIOMP.emitOMPTargetTeams(DSAValueMap, DL, Fn, BBEntry, StartBB, EndBB,
                                  AfterBB, TargetInfo,
                                  /* OMPLoopInfo */ nullptr,
                                  StructMappingInfoMap, IsDeviceTargetRegion);
      } else if (Dir == OMPD_target_enter_data) {
        if (IsDeviceTargetRegion)
          report_fatal_error("Target enter data should never appear inside a "
                             "device target region");

        CGIOMP.emitOMPTargetEnterData(Fn, BBEntry, DSAValueMap,
                                      StructMappingInfoMap);
      } else if (Dir == OMPD_target_exit_data) {
        if (IsDeviceTargetRegion)
          report_fatal_error("Target exit data should never appear inside a "
                             "device target region");

        CGIOMP.emitOMPTargetExitData(Fn, BBEntry, DSAValueMap,
                                      StructMappingInfoMap);
      } else if (Dir == OMPD_target_teams_distribute) {
        TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
        CGIOMP.emitOMPDistribute(DSAValueMap, StartBB, BBExit, OMPLoopInfo,
                                 /* IsStandalone */ false);
        CGIOMP.emitOMPTargetTeams(DSAValueMap, DL, Fn, BBEntry, StartBB, EndBB,
                                  AfterBB, TargetInfo, &OMPLoopInfo,
                                  StructMappingInfoMap, IsDeviceTargetRegion);
      } else if (Dir == OMPD_distribute_parallel_for) {
        CGIOMP.emitOMPDistributeParallelFor(DSAValueMap, StartBB, BBExit,
                                            OMPLoopInfo, ParRegionInfo,
                                            /* isStandalone */ false);
      } else if (Dir == OMPD_target_teams_distribute_parallel_for) {
        TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD;
        CGIOMP.emitOMPTargetTeamsDistributeParallelFor(
            DSAValueMap, DL, Fn, BBEntry, StartBB, EndBB, BBExit, AfterBB,
            OMPLoopInfo, ParRegionInfo, TargetInfo, StructMappingInfoMap,
            IsDeviceTargetRegion);
      } else {
        LLVM_DEBUG(dbgs() << "Unknown directive " << *CBEntry << "\n");
        assert(false && "Unknown directive");
      }

      if (verifyFunction(*Fn, &errs()))
        report_fatal_error("Verification of IntrinsicsOpenMP lowering failed!");
    }

    LLVM_DEBUG(dbgs() << "=== Dump Lowered Module\n"
                      << M << "=== End of Dump Lowered Module\n");

    LLVM_DEBUG(dbgs() << "=== End of IntrinsicsOpenMP pass\n");

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};
} // namespace

PreservedAnalyses IntrinsicsOpenMPPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  IntrinsicsOpenMP IOMP;
  bool Changed = IOMP.runOnModule(M);

  if (Changed)
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

char IntrinsicsOpenMP::ID = 0;
static RegisterPass<IntrinsicsOpenMP> X("intrinsics-openmp",
                                        "IntrinsicsOpenMP Pass");

ModulePass *llvm::createIntrinsicsOpenMPPass() {
  return new IntrinsicsOpenMP();
}

void LLVMAddIntrinsicsOpenMPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIntrinsicsOpenMPPass());
}
