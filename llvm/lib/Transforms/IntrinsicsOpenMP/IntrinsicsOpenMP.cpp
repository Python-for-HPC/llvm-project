//===- IntrinsicsOpenMP.cpp - Codegen OpenMP from IR intrinsics ---------------===//
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
#include "llvm/ADT/Statistic.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IntrinsicsOpenMP/IntrinsicsOpenMP.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

  // TODO: explose clauses through namespace omp?
  enum DSAType {
    DSA_PRIVATE,
    DSA_FIRSTPRIVATE,
    DSA_SHARED
  };

  static const DenseMap<StringRef, Directive> StringToDir = {
      {"DIR.OMP.PARALLEL", OMPD_parallel},
      {"DIR.OMP.SINGLE", OMPD_single},
      {"DIR.OMP.CRITICAL", OMPD_critical},
      {"DIR.OMP.BARRIER", OMPD_barrier},
      {"DIR.OMP.LOOP", OMPD_for},
      {"DIR.OMP.PARALLEL.LOOP", OMPD_parallel_for},
  };

  static const DenseMap<StringRef, DSAType> StringToDSA = {
      {"QUAL.OMP.PRIVATE", DSA_PRIVATE},
      {"QUAL.OMP.FIRSTPRIVATE", DSA_FIRSTPRIVATE},
      {"QUAL.OMP.SHARED", DSA_SHARED},
  };

  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  using BodyGenCallbackTy = OpenMPIRBuilder::BodyGenCallbackTy;
  using FinalizeCallbackTy = OpenMPIRBuilder::FinalizeCallbackTy;

  LoopInfo *LI = nullptr;
  DominatorTree *DT = nullptr;

  struct IntrinsicsOpenMP: public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    IntrinsicsOpenMP() : ModulePass(ID) {}

    static void emitOMPParallel(OpenMPIRBuilder &OMPBuilder,
                                DenseMap<Value *, DSAType> &DSAValueMap,
                                const DebugLoc &DL, Function *Fn,
                                BasicBlock *BBEntry, BasicBlock *AfterBB,
                                BodyGenCallbackTy BodyGenCB,
                                FinalizeCallbackTy FiniCB,
                                Value *IfCondition,
                                Value *NumThreads) {
      auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                        Value &Orig, Value &Inner,
                        Value *&ReplacementValue) -> InsertPointTy {
        auto It = DSAValueMap.find(&Orig);
        dbgs() << "DSAValueMap for Orig " << Orig << " Inner " << Inner;
        if (It != DSAValueMap.end())
          dbgs() << It->second;
        else
          dbgs() << " (null)!";
        dbgs() << "\n ";

        assert(It != DSAValueMap.end() && "Expected Value in DSAValueMap");

        if (It->second == DSA_PRIVATE) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, /*ArraySize */ nullptr, Inner.getName());
          dbgs() << "Privatizing Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n";
        } else if (It->second == DSA_FIRSTPRIVATE) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          Value *V = OMPBuilder.Builder.CreateLoad(VTy, &Inner,
                                                   Orig.getName() + ".reload");
          ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, /*ArraySize */ nullptr, Orig.getName() + ".copy");
          OMPBuilder.Builder.restoreIP(CodeGenIP);
          OMPBuilder.Builder.CreateStore(V, ReplacementValue);
          dbgs() << "Firstprivatizing Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n";
        } else {
          ReplacementValue = &Inner;
          dbgs() << "Shared Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n";
        }

        return CodeGenIP;
      };

      IRBuilder<>::InsertPoint AllocaIP(
          &Fn->getEntryBlock(), Fn->getEntryBlock().getFirstInsertionPt());

      // Set the insertion location at the end of the BBEntry.
      BBEntry->getTerminator()->eraseFromParent();

      Value *IfConditionCast = nullptr;
      if (IfCondition) {
        OMPBuilder.Builder.SetInsertPoint(BBEntry);
        IfConditionCast = OMPBuilder.Builder.CreateIntCast(
            IfCondition, OMPBuilder.Builder.getInt1Ty(), /* isSigned */ false);
      }

      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(BBEntry, BBEntry->end()), DL);

      // TODO: support cancellable, binding.
      InsertPointTy AfterIP = OMPBuilder.createParallel(
          Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
          /* IfCondition */ IfConditionCast, /* NumThreads */ NumThreads,
          OMP_PROC_BIND_default, /* IsCancellable */ false);
      BranchInst::Create(AfterBB, AfterIP.getBlock());
    }

    static void emitOMPFor(Module &M, OpenMPIRBuilder &OMPBuilder, Value *IV,
                           Value *UB, BasicBlock *PreHeader, BasicBlock *Exit,
                           OMPScheduleType Sched, Value *Chunk) {
      Type *IVTy = IV->getType()->getPointerElementType();

      auto GetKmpcForStaticInit = [&]() -> FunctionCallee {
        dbgs() << "Type " << *IVTy << "\n";
        unsigned Bitwidth = IVTy->getIntegerBitWidth();
        dbgs() << "Bitwidth " << Bitwidth << "\n";
        if (Bitwidth == 32)
          return OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_for_static_init_4u);
        if (Bitwidth == 64)
          return OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_for_static_init_8u);
        llvm_unreachable("unknown OpenMP loop iterator bitwidth");
      };
      // TODO: privatization.
      FunctionCallee KmpcForStaticInit = GetKmpcForStaticInit();
      FunctionCallee KmpcForStaticFini = OMPBuilder.getOrCreateRuntimeFunction(
          M, OMPRTL___kmpc_for_static_fini);

      const DebugLoc DL = PreHeader->getTerminator()->getDebugLoc();
      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(PreHeader, PreHeader->getTerminator()->getIterator()),
          DL);
      Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
      Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr);

      // Create allocas for static init values.
      InsertPointTy AllocaIP(PreHeader, PreHeader->getFirstInsertionPt());
      Type *I32Type = Type::getInt32Ty(M.getContext());
      OMPBuilder.Builder.restoreIP(AllocaIP);
      Value *PLastIter =
          OMPBuilder.Builder.CreateAlloca(I32Type, nullptr, "omp_lastiter");
      Value *PLowerBound =
          OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp_lb");
      Value *PStride =
          OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp_stride");
      Value *PUpperBound =
          OMPBuilder.Builder.CreateAlloca(IVTy, nullptr, "omp_ub");

      OMPBuilder.Builder.SetInsertPoint(PreHeader->getTerminator());

      // Store the initial normalized upper bound to PUpperBound.
      Value *LoadUB = OMPBuilder.Builder.CreateLoad(
          UB->getType()->getPointerElementType(), UB);
      OMPBuilder.Builder.CreateStore(LoadUB, PUpperBound);

      Constant *Zero = ConstantInt::get(IVTy, 0);
      Constant *One = ConstantInt::get(IVTy, 1);
      OMPBuilder.Builder.CreateStore(Zero, PLowerBound);
      OMPBuilder.Builder.CreateStore(One, PStride);

      // If Chunk is not specified (nullptr), default to one, complying with the
      // OpenMP specification.
      if (!Chunk)
        Chunk = One;
      Value *ChunkCast = OMPBuilder.Builder.CreateIntCast(Chunk, IVTy, /*isSigned*/ false);

      Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);

      // TODO: add more scheduling types.
      Constant *SchedulingType =
          ConstantInt::get(I32Type, static_cast<int>(Sched));

      dbgs() << "=== SchedulingType " << *SchedulingType << "\n";
      dbgs() << "=== PLowerBound " << *PLowerBound << "\n";
      dbgs() << "=== PUpperBound " << *PUpperBound << "\n";
      dbgs() << "=== PStride " << *PStride << "\n";
      dbgs() << "=== Incr " << *One << "\n";
      dbgs() << "=== ChunkCast " << *ChunkCast << "\n";
      OMPBuilder.Builder.CreateCall(
          KmpcForStaticInit, {SrcLoc, ThreadNum, SchedulingType, PLastIter,
                              PLowerBound, PUpperBound, PStride, One, ChunkCast});
      // Load returned upper bound to UB.
      Value *LoadPUpperBound = OMPBuilder.Builder.CreateLoad(
          PUpperBound->getType()->getPointerElementType(), PUpperBound);
      OMPBuilder.Builder.CreateStore(LoadPUpperBound, UB);
      // Add lower bound to IV.
      Value *LowerBound = OMPBuilder.Builder.CreateLoad(IVTy, PLowerBound);
      Value *LoadIV = OMPBuilder.Builder.CreateLoad(IVTy, IV);
      Value *UpdateIV = OMPBuilder.Builder.CreateAdd(LoadIV, LowerBound);
      OMPBuilder.Builder.CreateStore(UpdateIV, IV);

      // Add fini call to loop exit block.
      OMPBuilder.Builder.SetInsertPoint(Exit->getTerminator());
      OMPBuilder.Builder.CreateCall(KmpcForStaticFini, {SrcLoc, ThreadNum});

      // Create barrier in exit block.
      // TODO: only create if not in a combined parallel for directive.
      OMPBuilder.createBarrier(OpenMPIRBuilder::LocationDescription(
                                   OMPBuilder.Builder.saveIP(), Loc.DL),
                               omp::Directive::OMPD_for,
                               /* ForceSimpleCall */ false,
                               /* CheckCancelFlag */ false);
    }

    bool runOnModule(Module &M) override {
      dbgs() << "=== Start IntrinsicsOpenMPPass v4\n";

      Function *RegionEntryF = M.getFunction("llvm.directive.region.entry");

      // Return early for lack of directive intrinsics.
      if (!RegionEntryF) {
        dbgs() << "No intrinsics directives, exiting...\n";
        return false;
      }

      OpenMPIRBuilder OMPBuilder(M);
      OMPBuilder.initialize();

      dbgs() << "=== Dump module\n" << M << "=== End of Dump module\n";

      // Iterate over all calls to directive intrinsics and transform code
      // using OpenMPIRBuilder for lowering.
      SmallVector<User *, 4> RegionEntryUsers(RegionEntryF->users());
      for(User *Usr : RegionEntryUsers) {
        dbgs() << "Found Usr " << *Usr << "\n";
        CallBase *CBEntry = dyn_cast<CallBase>(Usr);
        assert(CBEntry && "Expected call to region entry intrinsic");

        // Extract the directive kind and data sharing attributes of values
        // from the operand bundles of the intrinsic call.
        Directive Dir = OMPD_unknown;
        SmallVector<OperandBundleDef, 16> OpBundles;
        DenseMap<Value *, DSAType>  DSAValueMap;

        struct {
          Value *IV = nullptr;
          Value *UB = nullptr;
          // Implementation defined: set default schedule to static.
          OMPScheduleType Sched = OMPScheduleType::Static;
          Value *Chunk = nullptr;
        } OMPLoopInfo;

        struct {
          Value *NumThreads = nullptr;
          Value *IfCondition = nullptr;
        } ParRegionInfo;

        CBEntry->getOperandBundlesAsDefs(OpBundles);
        // TODO: parse clauses.
        for(OperandBundleDef &O : OpBundles) {
          StringRef Tag = O.getTag();
          dbgs() << "OPB " << Tag << "\n";

          if (Tag.startswith("DIR")) {
            auto It = StringToDir.find(Tag);
            assert(It != StringToDir.end() && "Directive is not supported!");
            Dir = It->second;
          } else if (Tag.startswith("QUAL")) {
            for (auto I = O.input_begin(), E = O.input_end(); I != E; ++I) {
              Value *V = dyn_cast<Value>(*I);
              assert(V && "Expected Value");

              if (Tag.startswith("QUAL.OMP.NORMALIZED.IV")) {
                assert(O.input_size() == 1 && "Expected single IV value");
                OMPLoopInfo.IV = V;
              }
              else if (Tag.startswith("QUAL.OMP.NORMALIZED.UB")) {
                assert(O.input_size() == 1 && "Expected single UB value");
                OMPLoopInfo.UB = V;
              }
              else if (Tag.startswith("QUAL.OMP.NUM_THREADS")) {
                assert(O.input_size() == 1 && "Expected single NumThreads value");
                ParRegionInfo.NumThreads = V;
                // TODO: Check DSA value for NumThreads value.
                DSAValueMap[V] = DSA_FIRSTPRIVATE;
              }
              else if (Tag.startswith("QUAL.OMP.SCHEDULE")) {
                assert(O.input_size() == 1 && "Expected single chunking scheduling value");
                Constant *Zero = ConstantInt::get(V->getType(), 0);
                OMPLoopInfo.Chunk = V;

                if (Tag == "QUAL.OMP.SCHEDULE.STATIC") {
                  assert(V == Zero && "Chunking is not yet supported, requires "
                                      "the use of omp_stride (static_chunked)");
                  if (V == Zero)
                    OMPLoopInfo.Sched = OMPScheduleType::Static;
                  else
                    OMPLoopInfo.Sched = OMPScheduleType::StaticChunked;
                } else
                  assert(false && "Unsupported scheduling type");
              }
              else if (Tag.startswith("QUAL.OMP.IF")) {
                assert(O.input_size() == 1 && "Expected single if condition value");
                ParRegionInfo.IfCondition = V;
              }
              else /* DSA Qualifiers */ {
                auto It = StringToDSA.find(Tag);
                assert(It != StringToDSA.end() && "DSA type not found in map");
                DSAValueMap[V] = It->second;
              }
            }
          }
        }

        assert(Dir != OMPD_unknown && "Expected valid OMP directive");

        assert(CBEntry->getNumUses() == 1 &&
               "Expected single use of the directive entry CB");
        Use &U = *CBEntry->use_begin();
        CallBase *CBExit = dyn_cast<CallBase>(U.getUser());
        assert(CBExit && "Expected call to region exit intrinsic");
        dbgs() << "Found Use of " << *CBEntry << "\n-> AT ->\n" << *CBExit << "\n";

        // Gather info.
        BasicBlock *BBEntry = CBEntry->getParent();
        Function *Fn = BBEntry->getParent();
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();

        // Create the basic block structure to isolate the outlined region.
        BasicBlock *StartBB = SplitBlock(BBEntry, CBEntry, DT);
        assert(BBEntry->getUniqueSuccessor() == StartBB &&
               "Expected unique successor at region start BB");

        BasicBlock *BBExit = CBExit->getParent();
        BasicBlock *EndBB = SplitBlock(BBExit, CBExit->getNextNode(), DT);
        assert(BBExit->getUniqueSuccessor() == EndBB &&
               "Expected unique successor at region end BB");
        BasicBlock *AfterBB =
            SplitBlock(EndBB, &*EndBB->getFirstInsertionPt(), DT);

        // Define the default BodyGenCB lambda.
        auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                             BasicBlock &ContinuationIP) {
          BasicBlock *CGStartBB = CodeGenIP.getBlock();
          BasicBlock *CGEndBB =
              SplitBlock(CGStartBB, &*CodeGenIP.getPoint(), DT, LI);
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

        if(Dir == OMPD_parallel) {
          emitOMPParallel(OMPBuilder, DSAValueMap, DL, Fn, BBEntry, AfterBB,
                          BodyGenCB, FiniCB, ParRegionInfo.IfCondition,
                          ParRegionInfo.NumThreads);

          dbgs() << "=== Before Fn\n" << *Fn << "=== End of Before Fn\n";
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
          dbgs() << "=== Finalize Fn\n" << *Fn << "=== End of Finalize Fn\n";
        } else if (Dir == OMPD_single) {
          // Set the insertion location at the end of the BBEntry.
          BBEntry->getTerminator()->eraseFromParent();
          OpenMPIRBuilder::LocationDescription Loc(InsertPointTy(BBEntry, BBEntry->end()),
                                                   DL);

          InsertPointTy AfterIP = OMPBuilder.createSingle(
              Loc, BodyGenCB, FiniCB, /*DidIt*/ nullptr);
          BranchInst::Create(AfterBB, AfterIP.getBlock());
          dbgs() << "=== Single Fn\n" << *Fn << "=== End of Single Fn\n";
        } else if (Dir == OMPD_critical) {
          // Set the insertion location at the end of the BBEntry.
          BBEntry->getTerminator()->eraseFromParent();
          OpenMPIRBuilder::LocationDescription Loc(InsertPointTy(BBEntry, BBEntry->end()),
                                                   DL);

          InsertPointTy AfterIP = OMPBuilder.createCritical(Loc, BodyGenCB, FiniCB, "",
                                    /*HintInst*/ nullptr);
          BranchInst::Create(AfterBB, AfterIP.getBlock());
          dbgs() << "=== Critical Fn\n" << *Fn << "=== End of Critical Fn\n";
        }
        else if (Dir == OMPD_barrier) {
          // Set the insertion location at the end of the BBEntry.
          OpenMPIRBuilder::LocationDescription Loc(InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
                                                   DL);

          // TODO: check ForceSimpleCall usage.
          OMPBuilder.createBarrier(Loc, OMPD_barrier, /*ForceSimpleCall*/ false,
                                   /*CheckCancelFlag*/ true);
          dbgs() << "=== Barrier Fn\n" << *Fn << "=== End of Barrier Fn\n";
        }
        else if (Dir == OMPD_for) {
          dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n";
          dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n";
          assert(OMPLoopInfo.IV && "Expected non-null IV");
          assert(OMPLoopInfo.UB && "Expected non-null UB");

          BasicBlock *PreHeader = StartBB;
          BasicBlock *Header = PreHeader->getUniqueSuccessor();
          BasicBlock *Exit = EndBB;
          assert(Header &&
                 "Expected unique successor from PreHeader to Header");
          dbgs() << "=== PreHeader\n" << *PreHeader << "=== End of PreHeader\n";
          dbgs() << "=== Header\n" << *Header << "=== End of Header\n";
          dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n";

          emitOMPFor(M, OMPBuilder, OMPLoopInfo.IV, OMPLoopInfo.UB, PreHeader,
                     Exit, OMPLoopInfo.Sched, OMPLoopInfo.Chunk);
          dbgs() << "=== For Fn\n" << *Fn << "=== End of For Fn\n";
          //assert(false && "OMPD_for is not supported yet!");
        }
        else if (Dir == OMPD_parallel_for) {
          // TODO: Verify the DSA for IV, UB since they are implicit in the
          // combined directive entry.
          assert(OMPLoopInfo.IV && "Expected non-null IV");
          assert(OMPLoopInfo.UB && "Expected non-null UB");

          // TODO: Check setting DSA for IV, UB for correctness.
          DSAValueMap[OMPLoopInfo.IV] = DSA_PRIVATE;
          DSAValueMap[OMPLoopInfo.UB] = DSA_FIRSTPRIVATE;

          BasicBlock *PreHeader = StartBB;
          BasicBlock *Header = PreHeader->getUniqueSuccessor();
          BasicBlock *Exit = EndBB;
          assert(Header &&
                 "Expected unique successor from PreHeader to Header");
          dbgs() << "=== PreHeader\n" << *PreHeader << "=== End of PreHeader\n";
          dbgs() << "=== Header\n" << *Header << "=== End of Header\n";
          dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n";
          emitOMPFor(M, OMPBuilder, OMPLoopInfo.IV, OMPLoopInfo.UB, PreHeader,
                     Exit, OMPLoopInfo.Sched, OMPLoopInfo.Chunk);
          emitOMPParallel(OMPBuilder, DSAValueMap, DL, Fn, BBEntry, AfterBB,
                          BodyGenCB, FiniCB, ParRegionInfo.IfCondition, ParRegionInfo.NumThreads);
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
        }
        else {
          dbgs() << "Unknown directive " << *CBEntry << "\n";
          assert(false && "Unknown directive");
        }
      }

      dbgs() << "=== Dump Lowered Module\n" << M << "=== End of Dump Lowered Module\n";

      dbgs() << "=== End of IntrinscsOpenMP pass\n";
      return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      ModulePass::getAnalysisUsage(AU);
    }
  };

}

PreservedAnalyses IntrinsicsOpenMPPass::run(Module &M, ModuleAnalysisManager &AM) {
    IntrinsicsOpenMP IOMP;
    bool Changed = IOMP.runOnModule(M);

    if(Changed)
      return PreservedAnalyses::none();

    return PreservedAnalyses::all();
}

char IntrinsicsOpenMP::ID = 0;
static RegisterPass<IntrinsicsOpenMP> X("intrinsics-openmp", "IntrinsicsOpenMP Pass");

// TODO: Explicitly add the pass to the builder to make sure it runs before any
// optimization?
//static RegisterStandardPasses Y(PassManagerBuilder::EP_ModuleOptimizerEarly,
//                                [](const PassManagerBuilder &Builder,
//                                   legacy::PassManagerBase &PM) {
//                                  PM.add(new IntrinsicsOpenMP());
//                                });

//static RegisterStandardPasses Z(PassManagerBuilder::EP_EnabledOnOptLevel0,
//                                [](const PassManagerBuilder &Builder,
//                                   legacy::PassManagerBase &PM) {
//                                  PM.add(new IntrinsicsOpenMP());
//                                });
ModulePass *llvm::createIntrinsicsOpenMPPass() { return new IntrinsicsOpenMP(); }

void LLVMAddIntrinsicsOpenMPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIntrinsicsOpenMPPass());
}