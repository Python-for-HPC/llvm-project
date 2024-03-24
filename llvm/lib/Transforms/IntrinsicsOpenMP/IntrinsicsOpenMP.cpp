//===- IntrinsicsOpenMP.cpp - Codegen OpenMP from IR intrinsics
//--------------===//
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
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Dominators.h"
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
#include <memory>

using namespace llvm;
using namespace omp;
using namespace iomp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

class DirectiveRegion {
public:
  DirectiveRegion() = delete;

  void addNested(DirectiveRegion *DR) {
    // TODO: add into the nest, under the innermost nested directive.
  }

  const SmallVector<DirectiveRegion *, 4> &getNested() { return Nested; }

  CallBase *getEntry() { return CBEntry; }

  CallBase *getExit() { return CBExit; }

  void setParent(DirectiveRegion *P) { Parent = P; }

  const DirectiveRegion *getParent() { return Parent; }

  static DirectiveRegion *create(CallBase *CBEntry, CallBase *CBExit) {
    DirectiveRegion *DR = new DirectiveRegion(CBEntry, CBExit);
    return DR;
  }

private:
  CallBase *CBEntry;
  CallBase *CBExit;
  DirectiveRegion *Parent;
  SmallVector<DirectiveRegion *, 4> Nested;

  DirectiveRegion(CallBase *CBEntry, CallBase *CBExit)
      : CBEntry(CBEntry), CBExit(CBExit), Parent(this) {}
};

static SmallVector<Value *>
collectGlobalizedValues(DirectiveRegion &Directive) {

  SmallVector<Value *> GlobalizedValues;

  SmallVector<OperandBundleDef, 16> OpBundles;
  Directive.getEntry()->getOperandBundlesAsDefs(OpBundles);
  for (OperandBundleDef &O : OpBundles) {
    StringRef Tag = O.getTag();
    auto It = StringToDSA.find(Tag);
    if (It == StringToDSA.end())
      continue;

    const ArrayRef<Value *> &TagInputs = O.inputs();

    DSAType DSATy = It->second;

    switch (DSATy) {
    case iomp::DSA_FIRSTPRIVATE:
    case iomp::DSA_PRIVATE:
      continue;
    default:
      GlobalizedValues.push_back(TagInputs[0]);
    }
  }

  return GlobalizedValues;
}

struct IntrinsicsOpenMP : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  IntrinsicsOpenMP() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // Codegen for nested or combined constructs assumes code is generated
    // bottom-up, that is from the innermost directive to the outermost. This
    // simplifies handling of DSA attributes by avoiding renaming values (tags
    // contain pre-lowered values when defining the data sharing environment)
    // when an outlined function privatizes them in the DSAValueMap.
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
    // Find all calls to directive intrinsics.
    DenseMap<Function *, SmallVector<DirectiveRegion *, 4>>
        FunctionToDirectives;

    for (User *Usr : RegionEntryF->users()) {
      CallBase *CBEntry = dyn_cast<CallBase>(Usr);
      assert(CBEntry && "Expected call to directive entry");
      assert(CBEntry->getNumUses() == 1 &&
             "Expected single use of the directive entry");
      Use &U = *CBEntry->use_begin();
      CallBase *CBExit = dyn_cast<CallBase>(U.getUser());
      assert(CBExit && "Expected call to region exit intrinsic");
      Function *F = CBEntry->getFunction();
      assert(F == CBExit->getFunction() &&
             "Expected directive entry/exit in the same function");

      DirectiveRegion *DM = DirectiveRegion::create(CBEntry, CBExit);
      FunctionToDirectives[F].push_back(DM);
    }

    SmallVector<std::list<DirectiveRegion *>, 4> DirectiveListVector;
    // Create directive lists per function, list stores outermost to innermost.
    for (auto &FTD : FunctionToDirectives) {
      // Find the dominator tree for the function to find directive lists.
      DominatorTree DT(*FTD.getFirst());
      auto &DirectiveRegions = FTD.getSecond();

      // TODO: do we need a "tree" structure or are nesting lists enough?
#if 0
      for(auto *DR: DirectiveRegions) {
        // Skip directives for which parent is found.
        if (DR->getParent() != DR)
          continue;

        for(auto *IDR : DirectiveRegions) {
          if(IDR == DR)
            continue;

          // DR dominates IDR.
          if (DT.dominates(DR->getEntry(), IDR->getEntry()) &&
              DT.dominates(IDR->getExit(), DR->getExit())) {
                DR->addNested(IDR);
              }
        }
      }
#endif

      // First pass, sweep directive regions and form lists.
      for (auto *DR : DirectiveRegions) {
        bool Inserted = false;
        for (auto &DirectiveList : DirectiveListVector) {
          auto *Outer = DirectiveList.front();

          // If DR dominates the Outer directive then put it in front.
          if (DT.dominates(DR->getEntry(), Outer->getEntry()) &&
              DT.dominates(Outer->getExit(), DR->getExit())) {
            // XXX: modifies the iterator, should exit loop.
            DirectiveList.push_front(DR);
            Inserted = true;
            // Possibly merge with other lists now that Outer is updated to DR.
            for (auto &OtherDirectiveList : DirectiveListVector) {
              auto *Outer = OtherDirectiveList.front();
              if (Outer == DR)
                continue;

              if (DT.dominates(DR->getEntry(), Outer->getEntry()) &&
                  DT.dominates(Outer->getExit(), DR->getExit())) {
                DirectiveList.insert(DirectiveList.end(),
                                     OtherDirectiveList.begin(),
                                     OtherDirectiveList.end());
                OtherDirectiveList.clear();
              }
            }
            break;
          }

          // If DR is outside the Outer, continue.
          if (!(DT.dominates(Outer->getEntry(), DR->getEntry()) &&
                DT.dominates(DR->getExit(), Outer->getExit())))
            continue;

          // DR is inside the outer region, find where to put it in the
          // DirectiveList.
          auto InsertIt = DirectiveList.end();
          for (auto It = DirectiveList.begin(), End = DirectiveList.end();
               It != End; ++It) {
            DirectiveRegion *IDR = *It;
            // Insert it after the dominating directive.
            if (DT.dominates(IDR->getEntry(), DR->getEntry()) &&
                DT.dominates(DR->getExit(), IDR->getExit()))
              InsertIt = std::next(It);
          }

          // XXX: Modifies the iterator, should exit loop.
          DirectiveList.insert(InsertIt, DR);
          Inserted = true;
          break;
        }

        if (!Inserted)
          DirectiveListVector.push_back(std::list<DirectiveRegion *>{DR});
      }

      // Delete empty lists.
      DirectiveListVector.erase(
          std::remove_if(
              DirectiveListVector.begin(), DirectiveListVector.end(),
              [](std::list<DirectiveRegion *> &DL) { return DL.empty(); }),
          DirectiveListVector.end());
    }

    // Iterate all directive lists and codegen.
    for (auto &DirectiveList : DirectiveListVector) {
      // If the outermost directive is a TARGET directive, collect globalized
      // values to set for codegen.
      // TODO: implement Directives as a class, parse each directive before
      // codegen.
      auto *Outer = DirectiveList.front();
      if (Outer->getEntry()->getOperandBundleAt(0).getTagName().contains(
              "TARGET")) {
        auto GlobalizedValues = collectGlobalizedValues(*Outer);
        CGIOMP.setDeviceGlobalizedValues(GlobalizedValues);
      }
      // Iterate post-order, from innermost to outermost to avoid renaming
      // values in codegen.
      for (auto It = DirectiveList.rbegin(), E = DirectiveList.rend(); It != E;
           ++It) {
        DirectiveRegion *DR = *It;
        LLVM_DEBUG(dbgs() << "Found Directive" << *DR->getEntry() << "\n");
        // Extract the directive kind and data sharing attributes of values
        // from the operand bundles of the intrinsic call.
        Directive Dir = OMPD_unknown;
        SmallVector<OperandBundleDef, 16> OpBundles;
        DSAValueMapTy DSAValueMap;

        // RAII for directive metainfo structs.
        OMPLoopInfoStruct OMPLoopInfo;
        ParRegionInfoStruct ParRegionInfo;
        TargetInfoStruct TargetInfo;
        TeamsInfoStruct TeamsInfo;

        MapVector<Value *, SmallVector<FieldMappingInfo, 4>>
            StructMappingInfoMap;

        bool IsDeviceTargetRegion = false;

        DR->getEntry()->getOperandBundlesAsDefs(OpBundles);
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
              assert(O.input_size() == 1 &&
                     "Expected single if condition value");
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
              assert(O.input_size() == 1 &&
                     "Expected a single elf image string");
              ConstantDataArray *ELF =
                  dyn_cast<ConstantDataArray>(TagInputs[0]);
              assert(ELF && "Expected constant string for ELF");
              TargetInfo.ELF = ELF;
            } else if (Tag.startswith("QUAL.OMP.DEVICE")) {
              // TODO: Handle device selection for target regions.
            } else if (Tag.startswith("QUAL.OMP.NUM_TEAMS")) {
              assert(O.input_size() == 1 && "Expected single NumTeams value");
              switch (Dir) {
              case OMPD_target:
                TargetInfo.NumTeams = TagInputs[0];
                break;
              case OMPD_teams:
              case OMPD_teams_distribute:
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
                // report_fatal_error("Unsupported qualifier in directive");
                assert(false && "Unsupported qualifier in directive");
              }
            } else if (Tag.startswith("QUAL.OMP.THREAD_LIMIT")) {
              assert(O.input_size() == 1 &&
                     "Expected single ThreadLimit value");
              switch (Dir) {
              case OMPD_target:
                TargetInfo.ThreadLimit = TagInputs[0];
                break;
              case OMPD_teams:
              case OMPD_teams_distribute:
                TeamsInfo.ThreadLimit = TagInputs[0];
                break;
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.ThreadLimit = TagInputs[0];
                TeamsInfo.ThreadLimit = TagInputs[0];
                break;
              default:
                // report_fatal_error("Unsupported qualifier in directive");
                assert(false && "Unsupported qualifier in directive");
              }
            } else if (Tag.startswith("QUAL.OMP.NOWAIT")) {
              switch (Dir) {
              case OMPD_target:
              case OMPD_target_teams:
              case OMPD_target_teams_distribute:
              case OMPD_target_teams_distribute_parallel_for:
                TargetInfo.NoWait = true;
                break;
              default:
                assert(false && "Unsupported nowait qualifier in directive");
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

        // Gather info.
        BasicBlock *BBEntry = DR->getEntry()->getParent();
        Function *Fn = BBEntry->getParent();
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();

        // Create the basic block structure to isolate the outlined region.
        BasicBlock *StartBB = SplitBlock(BBEntry, DR->getEntry());
        assert(BBEntry->getUniqueSuccessor() == StartBB &&
               "Expected unique successor at region start BB");

        BasicBlock *BBExit = DR->getExit()->getParent();
        BasicBlock *EndBB = SplitBlock(BBExit, DR->getExit()->getNextNode());
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
        DR->getExit()->eraseFromParent();
        DR->getEntry()->eraseFromParent();

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
        } else if (Dir == OMPD_teams_distribute) {
          CGIOMP.emitOMPDistribute(DSAValueMap, StartBB, BBExit, OMPLoopInfo,
                                   /* IsStandalone */ false);
          CGIOMP.emitOMPTeams(DSAValueMap, nullptr, DL, Fn, BBEntry, StartBB,
                              EndBB, AfterBB, TeamsInfo);
        } else if (Dir == OMPD_target_teams) {
          TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
          CGIOMP.emitOMPTargetTeams(DSAValueMap, nullptr, DL, Fn, BBEntry,
                                    StartBB, EndBB, AfterBB, TargetInfo,
                                    /* OMPLoopInfo */ nullptr,
                                    StructMappingInfoMap, IsDeviceTargetRegion);
        } else if (Dir == OMPD_target_data) {
          if (IsDeviceTargetRegion)
            report_fatal_error("Target enter data should never appear inside a "
                               "device target region");
          CGIOMP.emitOMPTargetData(Fn, BBEntry, BBExit, DSAValueMap,
                                   StructMappingInfoMap);
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
        } else if (Dir == OMPD_target_update) {
          if (IsDeviceTargetRegion)
            report_fatal_error("Target exit data should never appear inside a "
                               "device target region");

          CGIOMP.emitOMPTargetUpdate(Fn, BBEntry, DSAValueMap,
                                     StructMappingInfoMap);
        } else if (Dir == OMPD_target_teams_distribute) {
          TargetInfo.ExecMode = OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_GENERIC;
          CGIOMP.emitOMPDistribute(DSAValueMap, StartBB, BBExit, OMPLoopInfo,
                                   /* IsStandalone */ false);
          CGIOMP.emitOMPTargetTeams(DSAValueMap, nullptr, DL, Fn, BBEntry,
                                    StartBB, EndBB, AfterBB, TargetInfo,
                                    &OMPLoopInfo, StructMappingInfoMap,
                                    IsDeviceTargetRegion);
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
          assert(false && "Unknown directive");
          report_fatal_error("Unknown directive");
        }

        if (verifyFunction(*Fn, &errs()))
          report_fatal_error(
              "Verification of IntrinsicsOpenMP lowering failed!");
      }
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
