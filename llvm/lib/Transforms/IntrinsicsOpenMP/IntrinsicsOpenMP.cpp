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
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/IntrinsicsOpenMP/IntrinsicsOpenMP.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "intrinsics-openmp"

// TODO: Increment.
STATISTIC(NumOpenMPRegions, "Counts number of OpenMP regions created");

namespace {

  // TODO: expose clauses through namespace omp?
  enum DSAType {
    DSA_PRIVATE,
    DSA_FIRSTPRIVATE,
    DSA_SHARED,
    DSA_REDUCTION_ADD,
    DSA_MAP_TO,
    DSA_MAP_FROM,
    DSA_MAP_TOFROM,
    DSA_MAP_TO_STRUCT,
    DSA_MAP_FROM_STRUCT,
    DSA_MAP_TOFROM_STRUCT,
    DSA_MAP_STRUCT
  };

  static const DenseMap<StringRef, Directive> StringToDir = {
      {"DIR.OMP.PARALLEL", OMPD_parallel},
      {"DIR.OMP.SINGLE", OMPD_single},
      {"DIR.OMP.CRITICAL", OMPD_critical},
      {"DIR.OMP.BARRIER", OMPD_barrier},
      {"DIR.OMP.LOOP", OMPD_for},
      {"DIR.OMP.PARALLEL.LOOP", OMPD_parallel_for},
      {"DIR.OMP.TASK", OMPD_task},
      {"DIR.OMP.TASKWAIT", OMPD_taskwait},
      {"DIR.OMP.TARGET", OMPD_target}
  };

  // TODO: add more reduction operators.
  static const DenseMap<StringRef, DSAType> StringToDSA = {
      {"QUAL.OMP.PRIVATE", DSA_PRIVATE},
      {"QUAL.OMP.FIRSTPRIVATE", DSA_FIRSTPRIVATE},
      {"QUAL.OMP.SHARED", DSA_SHARED},
      {"QUAL.OMP.REDUCTION.ADD", DSA_REDUCTION_ADD},
      {"QUAL.OMP.MAP.TO", DSA_MAP_TO},
      {"QUAL.OMP.MAP.FROM", DSA_MAP_FROM},
      {"QUAL.OMP.MAP.TOFROM", DSA_MAP_TOFROM},
      {"QUAL.OMP.MAP.TO.STRUCT", DSA_MAP_TO_STRUCT},
      {"QUAL.OMP.MAP.FROM.STRUCT", DSA_MAP_FROM_STRUCT},
      {"QUAL.OMP.MAP.TOFROM.STRUCT", DSA_MAP_TOFROM_STRUCT}
  };

  /// Data attributes for each data reference used in an OpenMP target region.
  enum tgt_map_type {
    // No flags
    OMP_TGT_MAPTYPE_NONE = 0x000,
    // copy data from host to device
    OMP_TGT_MAPTYPE_TO = 0x001,
    // copy data from device to host
    OMP_TGT_MAPTYPE_FROM = 0x002,
    // copy regardless of the reference count
    OMP_TGT_MAPTYPE_ALWAYS = 0x004,
    // force unmapping of data
    OMP_TGT_MAPTYPE_DELETE = 0x008,
    // map the pointer as well as the pointee
    OMP_TGT_MAPTYPE_PTR_AND_OBJ = 0x010,
    // pass device base address to kernel
    OMP_TGT_MAPTYPE_TARGET_PARAM = 0x020,
    // return base device address of mapped data
    OMP_TGT_MAPTYPE_RETURN_PARAM = 0x040,
    // private variable - not mapped
    OMP_TGT_MAPTYPE_PRIVATE = 0x080,
    // copy by value - not mapped
    OMP_TGT_MAPTYPE_LITERAL = 0x100,
    // mapping is implicit
    OMP_TGT_MAPTYPE_IMPLICIT = 0x200,
    // copy data to device
    OMP_TGT_MAPTYPE_CLOSE = 0x400,
    // runtime error if not already allocated
    OMP_TGT_MAPTYPE_PRESENT = 0x1000,
    // descriptor for non-contiguous target-update
    OMP_TGT_MAPTYPE_NON_CONTIG = 0x100000000000,
    // member of struct, member given by [16 MSBs] - 1
    OMP_TGT_MAPTYPE_MEMBER_OF = 0xffff000000000000
  };

  struct CGReduction {
    static OpenMPIRBuilder::InsertPointTy
    sumReduction(OpenMPIRBuilder::InsertPointTy IP, Value *LHS, Value *RHS,
                 Value *&Result) {
      IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
      Type *VTy = RHS->getType();
      if (VTy->isIntegerTy())
        Result = Builder.CreateAdd(LHS, RHS, "red.add");
      else if (VTy->isFloatTy() || VTy->isDoubleTy())
        Result = Builder.CreateFAdd(LHS, RHS, "red.add");
      else
        assert(false && "Unsupported type for sumReduction");
      return Builder.saveIP();
    }

    static OpenMPIRBuilder::InsertPointTy
    sumAtomicReduction(OpenMPIRBuilder::InsertPointTy IP, Value *LHS,
                       Value *RHS) {
      IRBuilder<> Builder(IP.getBlock(), IP.getPoint());
      Type *VTy = RHS->getType()->getPointerElementType();
      Value *Partial = Builder.CreateLoad(VTy, RHS, "red.partial");
      if (VTy->isIntegerTy())
        Builder.CreateAtomicRMW(AtomicRMWInst::Add, LHS, Partial, None,
                                AtomicOrdering::Monotonic);
      else if (VTy->isFloatTy() || VTy->isDoubleTy())
        Builder.CreateAtomicRMW(AtomicRMWInst::FAdd, LHS, Partial, None,
                                AtomicOrdering::Monotonic);
      else
        assert(false && "Unsupported type for sumAtomicReduction");
      return Builder.saveIP();
    }
  };

  struct OffloadingMappingArgsTy {
    Value *Sizes;
    Value *MapTypes;
    Value *MapNames;
    Value *BasePtrs;
    Value *Ptrs;
    size_t Size;
  };

  struct FieldMappingInfo {
    Value *Index;
    Value *Offset;
    Value *NumElements;
    DSAType MapType;
  };

  using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
  using BodyGenCallbackTy = OpenMPIRBuilder::BodyGenCallbackTy;
  using FinalizeCallbackTy = OpenMPIRBuilder::FinalizeCallbackTy;

  struct IntrinsicsOpenMP: public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    IntrinsicsOpenMP() : ModulePass(ID) {}

    StructType *TgtOffloadEntryTy = nullptr;

    static void emitOMPParallel(OpenMPIRBuilder &OMPBuilder,
                                MapVector<Value *, DSAType> &DSAValueMap,
                                const DebugLoc &DL, Function *Fn,
                                BasicBlock *BBEntry, BasicBlock *StartBB,
                                BasicBlock *EndBB, BasicBlock *AfterBB,
                                FinalizeCallbackTy FiniCB, Value *IfCondition,
                                Value *NumThreads) {
      InsertPointTy BodyIP, BodyAllocaIP;
      SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

      auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                        Value &Orig, Value &Inner,
                        Value *&ReplacementValue) -> InsertPointTy {
        auto It = DSAValueMap.find(&Orig);
        LLVM_DEBUG(dbgs() << "DSAValueMap for Orig " << Orig << " Inner " << Inner);
        if (It != DSAValueMap.end())
          LLVM_DEBUG(dbgs() << It->second);
        else
          LLVM_DEBUG(dbgs() << " (null)!");
        LLVM_DEBUG(dbgs() << "\n ");

        assert(It != DSAValueMap.end() && "Expected Value in DSAValueMap");

        DSAType DSA = It->second;

        if (DSA == DSA_PRIVATE) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, /*ArraySize */ nullptr, Inner.getName());
          OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                         ReplacementValue);
          LLVM_DEBUG(dbgs() << "Privatizing Inner " << Inner << " -> to -> "
                            << *ReplacementValue << "\n");
        } else if (DSA == DSA_FIRSTPRIVATE) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          Value *V = OMPBuilder.Builder.CreateLoad(VTy, &Inner,
                                                   Orig.getName() + ".reload");
          ReplacementValue = OMPBuilder.Builder.CreateAlloca(
              VTy, /*ArraySize */ nullptr, Orig.getName() + ".copy");
          OMPBuilder.Builder.restoreIP(CodeGenIP);
          OMPBuilder.Builder.CreateStore(V, ReplacementValue);
          LLVM_DEBUG(dbgs() << "Firstprivatizing Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n");
        } else if (DSA == DSA_REDUCTION_ADD) {
          OMPBuilder.Builder.restoreIP(AllocaIP);
          Type *VTy = Inner.getType()->getPointerElementType();
          Value *V = OMPBuilder.Builder.CreateAlloca(
              VTy, /* ArraySize */ nullptr, Orig.getName() + ".red.priv");
          ReplacementValue = V;

          OMPBuilder.Builder.restoreIP(CodeGenIP);
          // Store idempotent value based on operation and type.
          // TODO: create templated emitInitAndAppendInfo in CGReduction
          if (VTy->isIntegerTy())
            OMPBuilder.Builder.CreateStore(ConstantInt::get(VTy, 0), V);
          else if (VTy->isFloatTy() || VTy->isDoubleTy())
            OMPBuilder.Builder.CreateStore(ConstantFP::get(VTy, 0.0), V);
          else
            assert(false && "Unsupported type to init with idempotent reduction value");

          ReductionInfos.push_back({&Orig, V, CGReduction::sumReduction,
                                    CGReduction::sumAtomicReduction});

          return OMPBuilder.Builder.saveIP();
        } else {
          ReplacementValue = &Inner;
          LLVM_DEBUG(dbgs() << "Shared Inner " << Inner << " -> to -> "
                 << *ReplacementValue << "\n");
        }

        return CodeGenIP;
      };

      auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                           BasicBlock &ContinuationIP) {
        BasicBlock *CGStartBB = CodeGenIP.getBlock();
        BasicBlock *CGEndBB = SplitBlock(CGStartBB, &*CodeGenIP.getPoint());
        assert(StartBB != nullptr && "StartBB should not be null");
        CGStartBB->getTerminator()->setSuccessor(0, StartBB);
        assert(EndBB != nullptr && "EndBB should not be null");
        EndBB->getTerminator()->setSuccessor(0, CGEndBB);

        BodyIP = InsertPointTy(CGEndBB, CGEndBB->getFirstInsertionPt());
        BodyAllocaIP = AllocaIP;
      };

      IRBuilder<>::InsertPoint AllocaIP(
          &Fn->getEntryBlock(), Fn->getEntryBlock().getFirstInsertionPt());

      // Set the insertion location at the end of the BBEntry.
      BBEntry->getTerminator()->eraseFromParent();

      Value *IfConditionEval = nullptr;
      if (IfCondition) {
        OMPBuilder.Builder.SetInsertPoint(BBEntry);
        if (IfCondition->getType()->isFloatingPointTy())
          IfConditionEval = OMPBuilder.Builder.CreateFCmpUNE(
              IfCondition, ConstantFP::get(IfCondition->getType(), 0));
        else
          IfConditionEval = OMPBuilder.Builder.CreateICmpNE(
              IfCondition, ConstantInt::get(IfCondition->getType(), 0));
      }

      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(BBEntry, BBEntry->end()), DL);

      // TODO: support cancellable, binding.
      InsertPointTy AfterIP = OMPBuilder.createParallel(
          Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB,
          /* IfCondition */ IfConditionEval, /* NumThreads */ NumThreads,
          OMP_PROC_BIND_default, /* IsCancellable */ false);

      if (!ReductionInfos.empty())
        OMPBuilder.createReductions(BodyIP, BodyAllocaIP, ReductionInfos);

      BranchInst::Create(AfterBB, AfterIP.getBlock());
    }

    static void emitOMPFor(Module &M, OpenMPIRBuilder &OMPBuilder,
                           MapVector<Value *, DSAType> &DSAValueMap, Value *IV,
                           Value *UB, BasicBlock *PreHeader, BasicBlock *Exit,
                           OMPScheduleType Sched, Value *Chunk,
                           bool IsStandalone) {
      Type *IVTy = IV->getType()->getPointerElementType();
      SmallVector<OpenMPIRBuilder::ReductionInfo> ReductionInfos;

      auto GetKmpcForStaticInit = [&]() -> FunctionCallee {
        LLVM_DEBUG(dbgs() << "Type " << *IVTy << "\n");
        unsigned Bitwidth = IVTy->getIntegerBitWidth();
        LLVM_DEBUG(dbgs() << "Bitwidth " << Bitwidth << "\n");
        if (Bitwidth == 32)
          return OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_for_static_init_4u);
        if (Bitwidth == 64)
          return OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___kmpc_for_static_init_8u);
        llvm_unreachable("unknown OpenMP loop iterator bitwidth");
      };

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

      OpenMPIRBuilder::OutlineInfo OI;
      OI.EntryBB = PreHeader;
      OI.ExitBB = Exit;
      SmallPtrSet<BasicBlock *, 8> BlockSet;
      SmallVector<BasicBlock *, 8> BlockVector;
      OI.collectBlocks(BlockSet, BlockVector);

      // Do privatization if standalone.
      // TODO: create PrivCBHelper and re-use PrivCB from emitOMPParallel.
      if (IsStandalone)
        for (auto &It : DSAValueMap) {
          Value *Orig = It.first;
          DSAType DSA = It.second;
          Value *ReplacementValue = nullptr;
          Type *VTy = Orig->getType()->getPointerElementType();

          if (DSA == DSA_SHARED)
            continue;

          // Store previous uses to set them to the ReplacementValue after
          // privatization codegen.
          SetVector<Use *> Uses;
          for (Use &U : Orig->uses())
            if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
              if (BlockSet.count(UserI->getParent()))
                Uses.insert(&U);

          OMPBuilder.Builder.restoreIP(AllocaIP);
          if (DSA == DSA_PRIVATE) {
            ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                VTy, /*ArraySize */ nullptr, Orig->getName() + ".for.priv");
            OMPBuilder.Builder.CreateStore(Constant::getNullValue(VTy),
                                           ReplacementValue);
          } else if (DSA == DSA_FIRSTPRIVATE) {
            Value *V = OMPBuilder.Builder.CreateLoad(
                VTy, Orig, Orig->getName() + ".for.firstpriv.reload");
            ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                VTy, /*ArraySize */ nullptr,
                Orig->getName() + ".for.firstpriv.copy");
            OMPBuilder.Builder.CreateStore(V, ReplacementValue);
            // ReplacementValue = Orig;
          } else if (DSA == DSA_REDUCTION_ADD) {
            ReplacementValue = OMPBuilder.Builder.CreateAlloca(
                VTy, /* ArraySize */ nullptr, Orig->getName() + ".red.priv");

            // Store idempotent value based on operation and type.
            // TODO: create templated emitInitAndAppendInfo in CGReduction
            if (VTy->isIntegerTy())
              OMPBuilder.Builder.CreateStore(ConstantInt::get(VTy, 0),
                                             ReplacementValue);
            else if (VTy->isFloatTy() || VTy->isDoubleTy())
              OMPBuilder.Builder.CreateStore(ConstantFP::get(VTy, 0.0),
                                             ReplacementValue);
            else
              assert(
                  false &&
                  "Unsupported type to init with idempotent reduction value");

            ReductionInfos.push_back({Orig, ReplacementValue,
                                      CGReduction::sumReduction,
                                      CGReduction::sumAtomicReduction});
          } else
            assert(false && "Unsupported privatization");

          assert(ReplacementValue && "Expected non-null ReplacementValue");

          for (Use *UPtr : Uses)
            UPtr->set(ReplacementValue);
        }

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

      LLVM_DEBUG(dbgs() << "=== SchedulingType " << *SchedulingType << "\n");
      LLVM_DEBUG(dbgs() << "=== PLowerBound " << *PLowerBound << "\n");
      LLVM_DEBUG(dbgs() << "=== PUpperBound " << *PUpperBound << "\n");
      LLVM_DEBUG(dbgs() << "=== PStride " << *PStride << "\n");
      LLVM_DEBUG(dbgs() << "=== Incr " << *One << "\n");
      LLVM_DEBUG(dbgs() << "=== ChunkCast " << *ChunkCast << "\n");
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

      // Add fini call, reductions, and barrier after the loop exit block.
      BasicBlock *FiniBB = SplitBlock(Exit, &*Exit->getFirstInsertionPt());
      BasicBlock *NextFiniBB = SplitBlock(FiniBB, &*FiniBB->getFirstInsertionPt());
      OMPBuilder.Builder.SetInsertPoint(FiniBB, FiniBB->getFirstInsertionPt());
      OMPBuilder.Builder.CreateCall(KmpcForStaticFini, {SrcLoc, ThreadNum});

      // Emit reductions, barrier if standalone.
      if (IsStandalone) {
        if (!ReductionInfos.empty())
          OMPBuilder.createReductions(OMPBuilder.Builder.saveIP(), AllocaIP,
                                      ReductionInfos);

        OMPBuilder.Builder.SetInsertPoint(NextFiniBB->getTerminator());
        OMPBuilder.createBarrier(OpenMPIRBuilder::LocationDescription(
                                     OMPBuilder.Builder.saveIP(), Loc.DL),
                                 omp::Directive::OMPD_for,
                                 /* ForceSimpleCall */ false,
                                 /* CheckCancelFlag */ false);
      }
    }

    static void emitOMPTask(Module &M, OpenMPIRBuilder &OMPBuilder,
                            MapVector<Value *, DSAType> &DSAValueMap,
                            Function *Fn, BasicBlock *BBEntry,
                            BasicBlock *StartBB, BasicBlock *EndBB,
                            BasicBlock *AfterBB) {
      // Define types.
      // ************** START TYPE DEFINITION ************** //
      enum {
        TiedFlag = 0x1,
        FinalFlag = 0x2,
        DestructorsFlag = 0x8,
        PriorityFlag = 0x20,
        DetachableFlag = 0x40,
      };

      // This is a union for priority/firstprivate destructors, use the
      // routine entry pointer to allocate space since it is larger than
      // Int32Ty for priority, see kmp.h. Unused for now.
      StructType *KmpCmplrdataTy =
          StructType::create({OMPBuilder.TaskRoutineEntryPtr});
      StructType *KmpTaskTTy = StructType::create(
          {OMPBuilder.VoidPtr, OMPBuilder.TaskRoutineEntryPtr, OMPBuilder.Int32,
           KmpCmplrdataTy, KmpCmplrdataTy},
          "struct.kmp_task_t");
      Type *KmpTaskTPtrTy = KmpTaskTTy->getPointerTo();

      FunctionCallee KmpcOmpTaskAlloc = OMPBuilder.getOrCreateRuntimeFunction(
          M, OMPRTL___kmpc_omp_task_alloc);
      SmallVector<Type *, 8> SharedsTy;
      SmallVector<Type *, 8> PrivatesTy;
      for (auto &It : DSAValueMap) {
        Value *OriginalValue = It.first;
        if (It.second == DSA_SHARED)
          SharedsTy.push_back(OriginalValue->getType());
        else if (It.second == DSA_PRIVATE || It.second == DSA_FIRSTPRIVATE) {
          assert(isa<PointerType>(OriginalValue->getType()) &&
                 "Expected private, firstprivate value with pointer type");
          // Store a copy of the value, thus get the pointer element type.
          PrivatesTy.push_back(
              OriginalValue->getType()->getPointerElementType());
        } else
          assert(false && "Unknown DSA type");
      }

      StructType *KmpSharedsTTy = nullptr;
      if (SharedsTy.empty())
          KmpSharedsTTy = StructType::create(M.getContext(), "struct.kmp_shareds");
      else
          KmpSharedsTTy = StructType::create(SharedsTy, "struct.kmp_shareds");
      assert(KmpSharedsTTy && "Expected non-null KmpSharedsTTy");
      Type *KmpSharedsTPtrTy = KmpSharedsTTy->getPointerTo();
      StructType *KmpPrivatesTTy =
          StructType::create(PrivatesTy, "struct.kmp_privates");
      Type *KmpPrivatesTPtrTy = KmpPrivatesTTy->getPointerTo();
      StructType *KmpTaskTWithPrivatesTy = StructType::create(
          {KmpTaskTTy, KmpPrivatesTTy}, "struct.kmp_task_t_with_privates");
      Type *KmpTaskTWithPrivatesPtrTy = KmpTaskTWithPrivatesTy->getPointerTo();

      // Declare the task entry function.
      Function *TaskEntryFn = Function::Create(
          OMPBuilder.TaskRoutineEntry, GlobalValue::InternalLinkage,
          Fn->getAddressSpace(), Fn->getName() + ".omp_task_entry", &M);
      // Name arguments.
      TaskEntryFn->getArg(0)->setName(".global_tid");
      TaskEntryFn->getArg(1)->setName(".task_t_with_privates");

      // Declare the task outlined function.
      FunctionType *TaskOutlinedFnTy = FunctionType::get(
          OMPBuilder.Void,
          {OMPBuilder.Int32, OMPBuilder.Int32Ptr, OMPBuilder.VoidPtr,
           KmpTaskTPtrTy, KmpSharedsTPtrTy},
          /*isVarArg=*/false);
      Function *TaskOutlinedFn = Function::Create(
          TaskOutlinedFnTy, GlobalValue::InternalLinkage, Fn->getAddressSpace(),
          Fn->getName() + ".omp_task_outlined", &M);
      TaskOutlinedFn->getArg(0)->setName(".global_tid");
      TaskOutlinedFn->getArg(1)->setName(".part_id");
      TaskOutlinedFn->getArg(2)->setName(".privates");
      TaskOutlinedFn->getArg(3)->setName(".task.data");
      TaskOutlinedFn->getArg(4)->setName(".shareds");

      // ************** END TYPE DEFINITION ************** //

      // Emit kmpc_omp_task_alloc, kmpc_omp_task
      {
        const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
        OpenMPIRBuilder::LocationDescription Loc(
            InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
            DL);
        Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
        Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr);
        // TODO: parse clauses, for now fix flags to tied
        unsigned TaskFlags = TiedFlag;
        Value *SizeofShareds = nullptr;
        if (KmpSharedsTTy->isEmptyTy())
          SizeofShareds = OMPBuilder.Builder.getInt64(0);
        else
          SizeofShareds = OMPBuilder.Builder.getInt64(
              M.getDataLayout().getTypeAllocSize(KmpSharedsTTy));
        Value *SizeofKmpTaskTWithPrivates = OMPBuilder.Builder.getInt64(
            M.getDataLayout().getTypeAllocSize(KmpTaskTWithPrivatesTy));
        OMPBuilder.Builder.SetInsertPoint(BBEntry,
                                          BBEntry->getFirstInsertionPt());
        Value *ThreadNum = OMPBuilder.getOrCreateThreadID(SrcLoc);
        Value *KmpTaskTWithPrivatesVoidPtr = OMPBuilder.Builder.CreateCall(
            KmpcOmpTaskAlloc,
            {SrcLoc, ThreadNum, OMPBuilder.Builder.getInt32(TaskFlags),
             SizeofKmpTaskTWithPrivates, SizeofShareds, TaskEntryFn},
            ".task.data");
        Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
            KmpTaskTWithPrivatesVoidPtr, KmpTaskTWithPrivatesPtrTy);

        const unsigned KmpTaskTIdx = 0;
        const unsigned KmpSharedsIdx = 0;
        Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpTaskTIdx);
        Value *KmpSharedsGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTTy, KmpTaskT, KmpSharedsIdx);
        Value *KmpSharedsVoidPtr =
            OMPBuilder.Builder.CreateLoad(OMPBuilder.VoidPtr, KmpSharedsGEP);
        Value *KmpShareds = OMPBuilder.Builder.CreateBitCast(KmpSharedsVoidPtr,
                                                             KmpSharedsTPtrTy);
        const unsigned KmpPrivatesIdx = 1;
        Value *KmpPrivates = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, KmpPrivatesIdx);

        // Store shareds by reference, firstprivates by value, in task data storage.
        unsigned SharedsGEPIdx = 0;
        unsigned PrivatesGEPIdx = 0;
        for (auto &It : DSAValueMap) {
          Value *OriginalValue = It.first;
          if (It.second == DSA_SHARED) {
            Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpSharedsTTy, KmpShareds, SharedsGEPIdx,
                OriginalValue->getName() + ".task.shared");
            OMPBuilder.Builder.CreateStore(OriginalValue, SharedGEP);
            ++SharedsGEPIdx;
          } else if (It.second == DSA_FIRSTPRIVATE) {
            Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpPrivatesTTy, KmpPrivates, PrivatesGEPIdx,
                OriginalValue->getName() + ".task.firstprivate");
            Value *Load = OMPBuilder.Builder.CreateLoad(
                OriginalValue->getType()->getPointerElementType(),
                OriginalValue);
            OMPBuilder.Builder.CreateStore(Load, FirstprivateGEP);
            ++PrivatesGEPIdx;
          } else if (It.second == DSA_PRIVATE)
            ++PrivatesGEPIdx;
        }

        FunctionCallee KmpcOmpTask =
            OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_task);
        OMPBuilder.Builder.CreateCall(
            KmpcOmpTask, {SrcLoc, ThreadNum, KmpTaskTWithPrivatesVoidPtr});
      }

      // Emit task entry function.
      {
        BasicBlock *TaskEntryBB =
            BasicBlock::Create(M.getContext(), "entry", TaskEntryFn);
        OMPBuilder.Builder.SetInsertPoint(TaskEntryBB);
        const unsigned TaskTIdx= 0;
        const unsigned PrivatesIdx = 1;
        const unsigned SharedsIdx = 0;
        Value *GTId = TaskEntryFn->getArg(0);
        Value *KmpTaskTWithPrivates = OMPBuilder.Builder.CreateBitCast(
            TaskEntryFn->getArg(1), KmpTaskTWithPrivatesPtrTy);
        Value *KmpTaskT = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, TaskTIdx,
            ".task.data");
        Value *SharedsGEP = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTTy, KmpTaskT, SharedsIdx, ".shareds.gep");
        Value *SharedsVoidPtr = OMPBuilder.Builder.CreateLoad(
            OMPBuilder.VoidPtr, SharedsGEP, ".shareds.void.ptr");
        Value *Shareds = OMPBuilder.Builder.CreateBitCast(
            SharedsVoidPtr, KmpSharedsTPtrTy, ".shareds");

        Value *Privates = nullptr;
        if (PrivatesTy.empty()) {
          Privates = Constant::getNullValue(OMPBuilder.VoidPtr);
        } else {
          Value *PrivatesTyped = OMPBuilder.Builder.CreateStructGEP(
              KmpTaskTWithPrivatesTy, KmpTaskTWithPrivates, PrivatesIdx,
              ".privates");
          Privates = OMPBuilder.Builder.CreateBitCast(
              PrivatesTyped, OMPBuilder.VoidPtr, ".privates.void.ptr");
        }
        assert(Privates && "Expected non-null privates");

        const unsigned PartIdIdx = 2;
        Value *PartId = OMPBuilder.Builder.CreateStructGEP(
            KmpTaskTTy, KmpTaskT, PartIdIdx, ".part_id");
        OMPBuilder.Builder.CreateCall(
            TaskOutlinedFnTy, TaskOutlinedFn,
            {GTId, PartId, Privates, KmpTaskT, Shareds});
        OMPBuilder.Builder.CreateRet(OMPBuilder.Builder.getInt32(0));
      }

      // Emit TaskOutlinedFn code.
      {
        OpenMPIRBuilder::OutlineInfo OI;
        OI.EntryBB = StartBB;
        OI.ExitBB = EndBB;
        SmallPtrSet<BasicBlock *, 8> OutlinedBlockSet;
        SmallVector<BasicBlock *, 8> OutlinedBlockVector;
        OI.collectBlocks(OutlinedBlockSet, OutlinedBlockVector);
        BasicBlock *TaskOutlinedEntryBB =
            BasicBlock::Create(M.getContext(), "entry", TaskOutlinedFn);
        BasicBlock *TaskOutlinedExitBB =
            BasicBlock::Create(M.getContext(), "exit", TaskOutlinedFn);
        for (BasicBlock *BB : OutlinedBlockVector)
          BB->moveBefore(TaskOutlinedExitBB);
        // Explicitly move EndBB to the outlined functions, since OutlineInfo
        // does not contain it in the OutlinedBlockVector.
        EndBB->moveBefore(TaskOutlinedExitBB);
        EndBB->getTerminator()->setSuccessor(0, TaskOutlinedExitBB);

        OMPBuilder.Builder.SetInsertPoint(TaskOutlinedEntryBB);
        const unsigned KmpPrivatesArgNo = 2;
        const unsigned KmpSharedsArgNo = 4;
        Value *KmpPrivatesArgVoidPtr = TaskOutlinedFn->getArg(KmpPrivatesArgNo);
        Value *KmpPrivatesArg = OMPBuilder.Builder.CreateBitCast(
            KmpPrivatesArgVoidPtr, KmpPrivatesTPtrTy);
        Value *KmpSharedsArg = TaskOutlinedFn->getArg(KmpSharedsArgNo);

        // Replace shareds, privates, firstprivates to refer to task data
        // storage.
        unsigned SharedsGEPIdx = 0;
        unsigned PrivatesGEPIdx = 0;
        for (auto &It : DSAValueMap) {
          Value *OriginalValue = It.first;
          Value *ReplacementValue = nullptr;
          if (It.second == DSA_SHARED) {
            Value *SharedGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpSharedsTTy, KmpSharedsArg, SharedsGEPIdx,
                OriginalValue->getName() + ".task.shared.gep");
            ReplacementValue = OMPBuilder.Builder.CreateLoad(
                OriginalValue->getType(), SharedGEP,
                OriginalValue->getName() + ".task.shared");
            ++SharedsGEPIdx;
          } else if (It.second == DSA_PRIVATE) {
            Value *PrivateGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
                OriginalValue->getName() + ".task.private.gep");
            ReplacementValue = PrivateGEP;
            ++PrivatesGEPIdx;
          } else if (It.second == DSA_FIRSTPRIVATE) {
            Value *FirstprivateGEP = OMPBuilder.Builder.CreateStructGEP(
                KmpPrivatesTTy, KmpPrivatesArg, PrivatesGEPIdx,
                OriginalValue->getName() + ".task.firstprivate.gep");
            ReplacementValue = FirstprivateGEP;
            ++PrivatesGEPIdx;
          }
          else
            assert(false && "Unknown DSA type");

          assert(ReplacementValue && "Expected non-null ReplacementValue");
          SmallVector<User *, 8> Users(OriginalValue->users());
          for (User *U : Users)
            if (Instruction *I = dyn_cast<Instruction>(U))
              if (OutlinedBlockSet.contains(I->getParent()))
                I->replaceUsesOfWith(OriginalValue, ReplacementValue);
        }

        OMPBuilder.Builder.CreateBr(StartBB);
        OMPBuilder.Builder.SetInsertPoint(TaskOutlinedExitBB);
        OMPBuilder.Builder.CreateRetVoid();
        BBEntry->getTerminator()->setSuccessor(0, AfterBB);
      }
    }

    static void emitOMPOffloadingEntry(Module &M, OpenMPIRBuilder &OMPBuilder,
                                       StructType *TgtOffloadEntryTy,
                                       const Twine &DevFuncName,
                                       Value *EntryPtr,
                                       Constant *&OMPOffloadEntry) {

      Constant *DevFuncNameConstant =
          ConstantDataArray::getString(M.getContext(), DevFuncName.str());
      auto *GV = new GlobalVariable(
          M, DevFuncNameConstant->getType(),
          /* isConstant */ true, GlobalValue::InternalLinkage,
          DevFuncNameConstant, ".omp_offloading.entry_name", nullptr,
          GlobalVariable::NotThreadLocal,
          /* AddressSpace */ 0);
      GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

      Constant *EntryConst = dyn_cast<Constant>(EntryPtr);
      assert(EntryConst && "Expected constant entry pointer");
      OMPOffloadEntry =
          ConstantStruct::get(TgtOffloadEntryTy,
                              ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                                  EntryConst, OMPBuilder.VoidPtr),
                              ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                                  GV, OMPBuilder.Int8Ptr),
                              ConstantInt::get(OMPBuilder.SizeTy, 0),
                              ConstantInt::get(OMPBuilder.Int32, 0),
                              ConstantInt::get(OMPBuilder.Int32, 0));
      auto *OMPOffloadEntryGV = new GlobalVariable(
          M, TgtOffloadEntryTy,
          /* isConstant */ true, GlobalValue::WeakAnyLinkage, OMPOffloadEntry,
          ".omp_offloading.entry." + DevFuncName);
      OMPOffloadEntryGV->setSection("omp_offloading_entries");
      OMPOffloadEntryGV->setAlignment(Align(1));
    }

    void emitOMPOffloadingMappings(
        Module &M, OpenMPIRBuilder &OMPBuilder, InsertPointTy AllocaIP,
        MapVector<Value *, DSAType> &DSAValueMap,
        MapVector<Value *, SmallVector<FieldMappingInfo, 4>> &StructMappingInfoMap,
        OffloadingMappingArgsTy &OffloadingMappingArgs) {

      struct MapperInfo {
        Value *BasePtr;
        Value *Ptr;
        Value *Size;
      };

      SmallVector<MapperInfo, 8> MapperInfos;
      //SmallVector<Constant *, 8> OffloadSizes;
      SmallVector<Constant *, 8> OffloadMapTypes;
      SmallVector<Constant *, 8> OffloadMapNames;

      if (DSAValueMap.empty()) {
        OffloadingMappingArgs.BasePtrs = Constant::getNullValue(OMPBuilder.VoidPtrPtr);
        OffloadingMappingArgs.Ptrs = Constant::getNullValue(OMPBuilder.VoidPtrPtr);
        OffloadingMappingArgs.Sizes = Constant::getNullValue(OMPBuilder.Int64Ptr);
        OffloadingMappingArgs.MapTypes = Constant::getNullValue(OMPBuilder.Int64Ptr);
        OffloadingMappingArgs.MapNames = Constant::getNullValue(OMPBuilder.VoidPtrPtr);

        return;
      }

      auto EmitMappingEntry = [&](Value *Size, uint64_t MapType, Value *BasePtr,
                                  Value *Ptr) {
        OffloadMapTypes.push_back(ConstantInt::get(OMPBuilder.SizeTy, MapType));
        // TODO: maybe add debug info.
        OffloadMapNames.push_back(
            OMPBuilder.getOrCreateSrcLocStr(BasePtr->getName(), "", 0, 0));
        LLVM_DEBUG(dbgs() << "Emit mapping entry BasePtr " << *BasePtr
                          << " Ptr " << *Ptr << " Size " << *Size << " MapType "
                          << MapType << "\n");
        MapperInfos.push_back({BasePtr, Ptr, Size});
      };

      // Keep track of argument position, needed for struct mappings.
      for (auto &It : DSAValueMap) {
        Value *V = It.first;
        DSAType DSA = It.second;
        uint64_t MapType = 0;

        // Determine the map type, completely or partly (structs).
        switch (DSA) {
        case DSA_FIRSTPRIVATE:
          MapType = OMP_TGT_MAPTYPE_TARGET_PARAM | OMP_TGT_MAPTYPE_LITERAL;
          break;
        case DSA_MAP_TO:
          MapType = OMP_TGT_MAPTYPE_TARGET_PARAM | OMP_TGT_MAPTYPE_TO;
          break;
        case DSA_MAP_FROM:
          MapType = OMP_TGT_MAPTYPE_TARGET_PARAM | OMP_TGT_MAPTYPE_FROM;
          break;
        case DSA_MAP_TOFROM:
          MapType = OMP_TGT_MAPTYPE_TARGET_PARAM | OMP_TGT_MAPTYPE_TO |
                     OMP_TGT_MAPTYPE_FROM;
          break;
        case DSA_MAP_STRUCT:
        case DSA_PRIVATE:
          // do nothing
          break;
        default:
          assert(false && "Unknown mapping type");
          report_fatal_error("Unknown mapping type");
        }

        // Emit the mapping entry.
        Value *Size;
        switch (DSA) {
        case DSA_MAP_TO:
        case DSA_MAP_FROM:
        case DSA_MAP_TOFROM:
        case DSA_FIRSTPRIVATE:
          Size = ConstantInt::get(
              OMPBuilder.SizeTy,
              M.getDataLayout().getTypeAllocSize(V->getType()));
          EmitMappingEntry(Size, MapType, V, V);
          break;
        case DSA_MAP_STRUCT: {
          Size = ConstantInt::get(
              OMPBuilder.SizeTy, M.getDataLayout().getTypeAllocSize(
                                     V->getType()->getPointerElementType()));
          EmitMappingEntry(Size, OMP_TGT_MAPTYPE_TARGET_PARAM, V, V);
          // Stores the argument position (starting from 1) of the parent
          // struct, to be used to set MEMBER_OF in the map type.
          size_t ArgPos = MapperInfos.size();

          for (auto &FieldInfo : StructMappingInfoMap[V]) {
            switch (FieldInfo.MapType) {
              case DSA_MAP_TO_STRUCT:
                MapType = OMP_TGT_MAPTYPE_TO;
                break;
              case DSA_MAP_FROM_STRUCT:
                MapType = OMP_TGT_MAPTYPE_FROM;
                break;
              case DSA_MAP_TOFROM_STRUCT:
                MapType = OMP_TGT_MAPTYPE_TO | OMP_TGT_MAPTYPE_FROM;
                break;
              default:
                assert(false && "Unknown struct mapping type");
                report_fatal_error("Unknown struct mapping type");
            }
            // MEMBER_OF(Argument Position)
            const size_t MemberOfOffset = 48;
            uint64_t MemberOfBits = ArgPos << MemberOfOffset;
            uint64_t FieldMapType = MapType | MemberOfBits;
            auto *FieldGEP = OMPBuilder.Builder.CreateInBoundsGEP(
                V->getType()->getPointerElementType(), V,
                {OMPBuilder.Builder.getInt32(0), FieldInfo.Index});

            Value *BasePtr = nullptr;
            Value *Ptr = nullptr;

            if (FieldGEP->getType()->getPointerElementType()->isPointerTy()) {
              FieldMapType |= OMP_TGT_MAPTYPE_PTR_AND_OBJ;
              BasePtr = FieldGEP;
              auto *Load = OMPBuilder.Builder.CreateLoad(
                  BasePtr->getType()->getPointerElementType(), BasePtr);
              Ptr = OMPBuilder.Builder.CreateInBoundsGEP(
                  Load->getType()->getPointerElementType(), Load,
                  FieldInfo.Offset);
            }
            else {
              BasePtr = V;
              Ptr = OMPBuilder.Builder.CreateInBoundsGEP(
                  FieldGEP->getType()->getPointerElementType(), FieldGEP,
                  FieldInfo.Offset);
            }

            assert(BasePtr && "Expected non-null base pointer");
            assert(Ptr && "Expected non-null pointer");

            auto ElementSize =
                ConstantInt::get(OMPBuilder.SizeTy,
                                 M.getDataLayout().getTypeAllocSize(
                                     Ptr->getType()->getPointerElementType()));
            Value *NumElements = nullptr;

            // Load the value of NumElements if it is a pointer.
            if (FieldInfo.NumElements->getType()->isPointerTy())
              NumElements = OMPBuilder.Builder.CreateLoad(
                  OMPBuilder.SizeTy, FieldInfo.NumElements);
            else
              NumElements = FieldInfo.NumElements;

            auto *Size = OMPBuilder.Builder.CreateMul(ElementSize, NumElements);
            EmitMappingEntry(Size, FieldMapType, BasePtr, Ptr);
          }
          break;
        }
        case DSA_PRIVATE: {
          // do nothing
          break;
        }
        default:
          assert(false && "Unknown mapping type");
          report_fatal_error("Unknown mapping type");
        }
      }

      auto EmitConstantArrayGlobalBitCast =
          [&M, &OMPBuilder](SmallVectorImpl<Constant *> &Vector, Type *Ty,
                            Type *DestTy, StringRef Name) {
            auto *Init =
                ConstantArray::get(ArrayType::get(Ty, Vector.size()), Vector);
            auto *GV =
                new GlobalVariable(M, ArrayType::get(Ty, Vector.size()),
                                   /* isConstant */ true,
                                   GlobalVariable::PrivateLinkage, Init, Name);
            GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

            return OMPBuilder.Builder.CreateBitCast(GV, DestTy);
          };

      // TODO: offload_sizes can be a global of constants for optimization if all sizes
      // are constants.
      //OffloadingMappingArgs.Sizes =
      //    EmitConstantArrayGlobalBitCast(OffloadSizes, OMPBuilder.SizeTy,
      //                            OMPBuilder.Int64Ptr, ".offload_sizes");
      OffloadingMappingArgs.MapTypes =
          EmitConstantArrayGlobalBitCast(OffloadMapTypes, OMPBuilder.SizeTy,
                                  OMPBuilder.Int64Ptr, ".offload_maptypes");
      OffloadingMappingArgs.MapNames =
          EmitConstantArrayGlobalBitCast(OffloadMapNames, OMPBuilder.Int8Ptr,
                                  OMPBuilder.VoidPtrPtr, ".offload_mapnames");

      auto EmitArrayAlloca =
          [&M, &OMPBuilder, &AllocaIP](size_t Size, Type *Ty, StringRef Name) {
            InsertPointTy CodeGenIP = OMPBuilder.Builder.saveIP();

            OMPBuilder.Builder.restoreIP(AllocaIP);
            auto *Alloca = OMPBuilder.Builder.CreateAlloca(
                ArrayType::get(Ty, Size), nullptr, Name);

            OMPBuilder.Builder.restoreIP(CodeGenIP);

            return Alloca;
          };

      auto *BasePtrsAlloca =
          EmitArrayAlloca(MapperInfos.size(), OMPBuilder.VoidPtr, ".offload_baseptrs");
      auto *PtrsAlloca =
          EmitArrayAlloca(MapperInfos.size(), OMPBuilder.VoidPtr, ".offload_ptrs");
      auto *SizesAlloca =
          EmitArrayAlloca(MapperInfos.size(), OMPBuilder.SizeTy, ".offload_sizes");

      size_t Idx = 0;
      for (auto &MI : MapperInfos) {
        // Store in the base pointers alloca.
        auto *GEP = OMPBuilder.Builder.CreateInBoundsGEP(
            BasePtrsAlloca->getType()->getPointerElementType(), BasePtrsAlloca,
            {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
        auto *Bitcast = OMPBuilder.Builder.CreateBitCast(
            GEP, MI.BasePtr->getType()->getPointerTo());
        OMPBuilder.Builder.CreateStore(MI.BasePtr, Bitcast);

        // Store in the pointers alloca.
        GEP = OMPBuilder.Builder.CreateInBoundsGEP(
            PtrsAlloca->getType()->getPointerElementType(), PtrsAlloca,
            {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
        Bitcast = OMPBuilder.Builder.CreateBitCast(
            GEP, MI.Ptr->getType()->getPointerTo());
        OMPBuilder.Builder.CreateStore(MI.Ptr, Bitcast);

        // Store in the sizes alloca.
        GEP = OMPBuilder.Builder.CreateInBoundsGEP(
            SizesAlloca->getType()->getPointerElementType(), SizesAlloca,
            {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(Idx)});
        Bitcast = OMPBuilder.Builder.CreateBitCast(
            GEP, MI.Size->getType()->getPointerTo());
        OMPBuilder.Builder.CreateStore(MI.Size, Bitcast);

        Idx++;
      }

      OffloadingMappingArgs.Size = MapperInfos.size();
      OffloadingMappingArgs.BasePtrs = OMPBuilder.Builder.CreateBitCast(
          BasePtrsAlloca, OMPBuilder.VoidPtrPtr);
      OffloadingMappingArgs.Ptrs =
          OMPBuilder.Builder.CreateBitCast(PtrsAlloca, OMPBuilder.VoidPtrPtr);
      OffloadingMappingArgs.Sizes = OMPBuilder.Builder.CreateBitCast(
          SizesAlloca, OMPBuilder.SizeTy->getPointerTo());

      //OffloadingMappingArgs.BasePtrs = OMPBuilder.Builder.CreateInBoundsGEP(
      //    BasePtrsAlloca->getType()->getPointerElementType(), BasePtrsAlloca,
      //    {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
      //OffloadingMappingArgs.Ptrs = OMPBuilder.Builder.CreateInBoundsGEP(
      //    PtrsAlloca->getType()->getPointerElementType(), PtrsAlloca,
      //    {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
      //OffloadingMappingArgs.Sizes = OMPBuilder.Builder.CreateInBoundsGEP(
      //    SizesAlloca->getType()->getPointerElementType(), SizesAlloca,
      //    {OMPBuilder.Builder.getInt32(0), OMPBuilder.Builder.getInt32(0)});
    }

    bool runOnModule(Module &M) override {
      LLVM_DEBUG(dbgs() << "=== Start IntrinsicsOpenMPPass v4\n");

      Function *RegionEntryF = M.getFunction("llvm.directive.region.entry");

      // Return early for lack of directive intrinsics.
      if (!RegionEntryF) {
        LLVM_DEBUG(dbgs() << "No intrinsics directives, exiting...\n");
        return false;
      }

      OpenMPIRBuilder OMPBuilder(M);
      OMPBuilder.initialize();

      // Types
      // TODO: fix naming, why .0 is added at the end?
      if (!TgtOffloadEntryTy) {
        TgtOffloadEntryTy = StructType::create(
            {OMPBuilder.Int8Ptr, OMPBuilder.Int8Ptr, OMPBuilder.SizeTy,
             OMPBuilder.Int32, OMPBuilder.Int32},
            "struct.__tgt_offload_entry");
      }
      // Outer scoped variables: TODO fix
      GlobalVariable *OMPOffloadEntries = nullptr;

      LLVM_DEBUG(dbgs() << "=== Dump module\n" << M << "=== End of Dump module\n");

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
        MapVector<Value *, DSAType> DSAValueMap;

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

        MapVector<Value *, SmallVector<FieldMappingInfo, 4>> StructMappingInfoMap;
        // TODO: Fix scoping under an info struct.
        GlobalVariable *OMPRegionId;

        bool IsOpenMPDevice = false;

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
            } else if (Tag.startswith("QUAL.OMP.NORMALIZED.UB")) {
              assert(O.input_size() == 1 && "Expected single UB value");
              OMPLoopInfo.UB = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.NUM_THREADS")) {
              assert(O.input_size() == 1 && "Expected single NumThreads value");
              ParRegionInfo.NumThreads = TagInputs[0];
            } else if (Tag.startswith("QUAL.OMP.SCHEDULE")) {
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
              DSAValueMap[TagInputs[0]] = DSA_REDUCTION_ADD;
            } else if (Tag.startswith("QUAL.OMP.TARGET.DEV_FUNC")) {
              assert(O.input_size() == 1 &&
                     "Expected a single device function name");
              ConstantDataArray *DevFuncArray =
                  dyn_cast<ConstantDataArray>(TagInputs[0]);
              assert(DevFuncArray &&
                     "Expected constant string for the device function");
              Twine DevFuncName =
                  "__omp_offload_numba_" + DevFuncArray->getAsString();

              // TODO: assumes 1 target region, can we call tgt_register_lib
              // multiple times?
              OMPRegionId = new GlobalVariable(
                  M, OMPBuilder.Int8, /* isConstant */ true,
                  GlobalValue::WeakAnyLinkage,
                  ConstantInt::get(OMPBuilder.Int8, 0),
                  DevFuncName + ".region_id", nullptr,
                  GlobalVariable::NotThreadLocal, /* AddressSpace */ 0);

              Constant *OMPOffloadEntry;
              emitOMPOffloadingEntry(M, OMPBuilder, TgtOffloadEntryTy,
                                     DevFuncName, OMPRegionId, OMPOffloadEntry);

              // TODO: do this at finalization when all entries have been
              // found.
              // TODO: assumes 1 device image, can we call tgt_register_lib
              // multiple times?
              auto *ArrayTy = ArrayType::get(TgtOffloadEntryTy, 1);
              OMPOffloadEntries = new GlobalVariable(
                  M, ArrayTy,
                  /* isConstant */ true, GlobalValue::ExternalLinkage,
                  ConstantArray::get(ArrayTy, {OMPOffloadEntry}),
                  ".omp_offloading.entries");

            } else if (Tag.startswith("QUAL.OMP.TARGET.ELF")) {
              assert(O.input_size() == 1 &&
                     "Expected a single elf image string");
              ConstantDataArray *ELFConstant =
                  dyn_cast<ConstantDataArray>(TagInputs[0]);
              assert(ELFConstant && "Expected constant string for ELF");
              auto *GV = new GlobalVariable(
                  M, ELFConstant->getType(), /* isConstant */ true,
                  GlobalValue::InternalLinkage, ELFConstant,
                  ".omp_offloading.device_image");
              GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

              StructType *TgtDeviceImageTy =
                  StructType::create({OMPBuilder.Int8Ptr, OMPBuilder.Int8Ptr,
                                      TgtOffloadEntryTy->getPointerTo(),
                                      TgtOffloadEntryTy->getPointerTo()},
                                     "struct.__tgt_device_image");

              StructType *TgtBinDescTy = StructType::create(
                  {OMPBuilder.Int32, TgtDeviceImageTy->getPointerTo(),
                   TgtOffloadEntryTy->getPointerTo(),
                   TgtOffloadEntryTy->getPointerTo()},
                  "struct.__tgt_bin_desc");

              auto *ArrayTy = ArrayType::get(TgtDeviceImageTy, 1);
              auto *Zero = ConstantInt::get(OMPBuilder.SizeTy, 0);
              auto *One = ConstantInt::get(OMPBuilder.SizeTy, 1);
              auto *Size = ConstantInt::get(OMPBuilder.SizeTy,
                                            ELFConstant->getNumElements());
              Constant *ZeroZero[] = {Zero, Zero};
              Constant *ZeroOne[] = {Zero, One};
              Constant *ZeroSize[] = {Zero, Size};

              auto *ImageB = ConstantExpr::getGetElementPtr(GV->getValueType(),
                                                            GV, ZeroZero);
              auto *ImageE = ConstantExpr::getGetElementPtr(GV->getValueType(),
                                                            GV, ZeroSize);
              auto *EntriesB = ConstantExpr::getGetElementPtr(
                  OMPOffloadEntries->getValueType(), OMPOffloadEntries,
                  ZeroZero);
              auto *EntriesE = ConstantExpr::getGetElementPtr(
                  OMPOffloadEntries->getValueType(), OMPOffloadEntries,
                  ZeroOne);

              auto *DeviceImageEntry = ConstantStruct::get(
                  TgtDeviceImageTy, ImageB, ImageE, EntriesB, EntriesE);
              auto *DeviceImages = new GlobalVariable(
                  M, ArrayTy,
                  /* isConstant */ true, GlobalValue::InternalLinkage,
                  ConstantArray::get(ArrayTy, {DeviceImageEntry}),
                  ".omp_offloading.device_images");

              auto *ImagesB = ConstantExpr::getGetElementPtr(
                  DeviceImages->getValueType(), DeviceImages, ZeroZero);
              auto *DescInit = ConstantStruct::get(
                  TgtBinDescTy,
                  ConstantInt::get(OMPBuilder.Int32,
                                   /* number of images */ 1),
                  ImagesB, EntriesB, EntriesE);
              auto *BinDesc = new GlobalVariable(
                  M, DescInit->getType(),
                  /* isConstant */ true, GlobalValue::InternalLinkage, DescInit,
                  ".omp_offloading.descriptor");

              // Add tgt_register_requires, tgt_register_lib,
              // tgt_unregister_lib.
              {
                // tgt_register_requires.
                auto *FuncTy =
                    FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
                auto *Func =
                    Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                     ".omp_offloading.requires_reg", &M);
                Func->setSection(".text.startup");

                // Get __tgt_register_lib function declaration.
                auto *RegFuncTy =
                    FunctionType::get(OMPBuilder.Void, OMPBuilder.Int64,
                                      /*isVarArg*/ false);
                FunctionCallee RegFuncC =
                    M.getOrInsertFunction("__tgt_register_requires", RegFuncTy);

                // Construct function body
                IRBuilder<> Builder(
                    BasicBlock::Create(M.getContext(), "entry", Func));
                // TODO: fix to pass the requirements enum value.
                Builder.CreateCall(RegFuncC,
                                   ConstantInt::get(OMPBuilder.Int64, 1));
                Builder.CreateRetVoid();

                // Add this function to constructors.
                // Set priority to 1 so that __tgt_register_lib is executed
                // AFTER
                // __tgt_register_requires (we want to know what requirements
                // have been asked for before we load a libomptarget plugin so
                // that by the time the plugin is loaded it can report how
                // many devices there are which can satisfy these
                // requirements).
                appendToGlobalCtors(M, Func, /*Priority*/ 0);
              }
              {
                // ctor
                auto *FuncTy =
                    FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
                auto *Func =
                    Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                     ".omp_offloading.descriptor_reg", &M);
                Func->setSection(".text.startup");

                // Get __tgt_register_lib function declaration.
                auto *RegFuncTy = FunctionType::get(
                    OMPBuilder.Void, TgtBinDescTy->getPointerTo(),
                    /*isVarArg*/ false);
                FunctionCallee RegFuncC =
                    M.getOrInsertFunction("__tgt_register_lib", RegFuncTy);

                // Construct function body
                IRBuilder<> Builder(
                    BasicBlock::Create(M.getContext(), "entry", Func));
                Builder.CreateCall(RegFuncC, BinDesc);
                Builder.CreateRetVoid();

                // Add this function to constructors.
                // Set priority to 1 so that __tgt_register_lib is executed
                // AFTER
                // __tgt_register_requires (we want to know what requirements
                // have been asked for before we load a libomptarget plugin so
                // that by the time the plugin is loaded it can report how
                // many devices there are which can satisfy these
                // requirements).
                appendToGlobalCtors(M, Func, /*Priority*/ 1);
              }
              {
                auto *FuncTy =
                    FunctionType::get(OMPBuilder.Void, /*isVarArg*/ false);
                auto *Func =
                    Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                     ".omp_offloading.descriptor_unreg", &M);
                Func->setSection(".text.startup");

                // Get __tgt_unregister_lib function declaration.
                auto *UnRegFuncTy = FunctionType::get(
                    OMPBuilder.Void, TgtBinDescTy->getPointerTo(),
                    /*isVarArg*/ false);
                FunctionCallee UnRegFuncC =
                    M.getOrInsertFunction("__tgt_unregister_lib", UnRegFuncTy);

                // Construct function body
                IRBuilder<> Builder(
                    BasicBlock::Create(M.getContext(), "entry", Func));
                Builder.CreateCall(UnRegFuncC, BinDesc);
                Builder.CreateRetVoid();

                // Add this function to global destructors.
                // Match priority of __tgt_register_lib
                appendToGlobalDtors(M, Func, /*Priority*/ 1);
              }
            } else if (Tag.startswith("QUAL.OMP.DEVICE")) {
              // TODO: Handle device selection for target regions.
            } else /* DSA Qualifiers */ {
              auto It = StringToDSA.find(Tag);
              assert(It != StringToDSA.end() && "DSA type not found in map");
              if (It->second == DSA_MAP_TO_STRUCT ||
                  It->second == DSA_MAP_FROM_STRUCT ||
                  It->second == DSA_MAP_TOFROM_STRUCT) {
                assert((TagInputs.size() - 1) == 3 &&
                       "Expected input triple for struct mapping");
                Value *Index = TagInputs[1];
                Value *Offset = TagInputs[2];
                Value *NumElements = TagInputs[3];
                StructMappingInfoMap[TagInputs[0]].push_back(
                    {Index, Offset, NumElements, It->second});

                DSAValueMap[TagInputs[0]] = DSA_MAP_STRUCT;
              }
              else
                DSAValueMap[TagInputs[0]] = It->second;
            }
          } else {
            // TODO: remove special handler for OMP.DEVICE, make it a qualifier,
            // error-check that OPBs are recognized.
            IsOpenMPDevice = true;
          }
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
          emitOMPParallel(OMPBuilder, DSAValueMap, DL, Fn, BBEntry, StartBB,
                          EndBB, AfterBB, FiniCB,
                          ParRegionInfo.IfCondition, ParRegionInfo.NumThreads);

          LLVM_DEBUG(dbgs() << "=== Before Fn\n" << *Fn << "=== End of Before Fn\n");
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
          LLVM_DEBUG(dbgs() << "=== Finalize Fn\n" << *Fn << "=== End of Finalize Fn\n");
        } else if (Dir == OMPD_single) {
          // Set the insertion location at the end of the BBEntry.
          BBEntry->getTerminator()->eraseFromParent();
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->end()), DL);

          InsertPointTy AfterIP = OMPBuilder.createSingle(
              Loc, BodyGenCB, FiniCB, /*DidIt*/ nullptr);
          BranchInst::Create(AfterBB, AfterIP.getBlock());
          LLVM_DEBUG(dbgs() << "=== Single Fn\n" << *Fn << "=== End of Single Fn\n");
        } else if (Dir == OMPD_critical) {
          // Set the insertion location at the end of the BBEntry.
          BBEntry->getTerminator()->eraseFromParent();
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->end()), DL);

          InsertPointTy AfterIP =
              OMPBuilder.createCritical(Loc, BodyGenCB, FiniCB, "",
                                        /*HintInst*/ nullptr);
          BranchInst::Create(AfterBB, AfterIP.getBlock());
          LLVM_DEBUG(dbgs() << "=== Critical Fn\n" << *Fn << "=== End of Critical Fn\n");
        } else if (Dir == OMPD_barrier) {
          // Set the insertion location at the end of the BBEntry.
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
              DL);

          // TODO: check ForceSimpleCall usage.
          OMPBuilder.createBarrier(Loc, OMPD_barrier,
                                   /*ForceSimpleCall*/ false,
                                   /*CheckCancelFlag*/ true);
          LLVM_DEBUG(dbgs() << "=== Barrier Fn\n" << *Fn << "=== End of Barrier Fn\n");
        } else if (Dir == OMPD_for) {
          LLVM_DEBUG(dbgs() << "OMPLoopInfo.IV " << *OMPLoopInfo.IV << "\n");
          LLVM_DEBUG(dbgs() << "OMPLoopInfo.UB " << *OMPLoopInfo.UB << "\n");
          assert(OMPLoopInfo.IV && "Expected non-null IV");
          assert(OMPLoopInfo.UB && "Expected non-null UB");

          BasicBlock *PreHeader = StartBB;
          BasicBlock *Header = PreHeader->getUniqueSuccessor();
          BasicBlock *Exit = BBExit;
          assert(Header &&
                 "Expected unique successor from PreHeader to Header");
          LLVM_DEBUG(dbgs() << "=== PreHeader\n" << *PreHeader << "=== End of PreHeader\n");
          LLVM_DEBUG(dbgs() << "=== Header\n" << *Header << "=== End of Header\n");
          LLVM_DEBUG(dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n");

          emitOMPFor(M, OMPBuilder, DSAValueMap, OMPLoopInfo.IV, OMPLoopInfo.UB,
                     PreHeader, Exit, OMPLoopInfo.Sched, OMPLoopInfo.Chunk,
                     /* IsStandalone */ true);
          LLVM_DEBUG(dbgs() << "=== For Fn\n" << *Fn << "=== End of For Fn\n");
        } else if (Dir == OMPD_parallel_for) {
          // TODO: Verify the DSA for IV, UB since they are implicit in the
          // combined directive entry.
          assert(OMPLoopInfo.IV && "Expected non-null IV");
          assert(OMPLoopInfo.UB && "Expected non-null UB");

          // TODO: Check setting DSA for IV, UB for correctness.
          DSAValueMap[OMPLoopInfo.IV] = DSA_PRIVATE;
          DSAValueMap[OMPLoopInfo.UB] = DSA_FIRSTPRIVATE;

          BasicBlock *PreHeader = StartBB;
          BasicBlock *Header = PreHeader->getUniqueSuccessor();
          BasicBlock *Exit = BBExit;
          assert(Header &&
                 "Expected unique successor from PreHeader to Header");
          LLVM_DEBUG(dbgs() << "=== PreHeader\n" << *PreHeader << "=== End of PreHeader\n");
          LLVM_DEBUG(dbgs() << "=== Header\n" << *Header << "=== End of Header\n");
          LLVM_DEBUG(dbgs() << "=== Exit \n" << *Exit << "=== End of Exit\n");
          emitOMPFor(M, OMPBuilder, DSAValueMap, OMPLoopInfo.IV, OMPLoopInfo.UB, PreHeader,
                     Exit, OMPLoopInfo.Sched, OMPLoopInfo.Chunk, /* IsStandalone */ false);
          emitOMPParallel(OMPBuilder, DSAValueMap, DL, Fn, BBEntry, StartBB,
                          EndBB, AfterBB, FiniCB,
                          ParRegionInfo.IfCondition, ParRegionInfo.NumThreads);
          OMPBuilder.finalize(Fn, /* AllowExtractorSinking */ true);
        } else if (Dir == OMPD_task) {
          emitOMPTask(M, OMPBuilder, DSAValueMap, Fn, BBEntry, StartBB, EndBB,
                      AfterBB);
        } else if (Dir == OMPD_taskwait) {
          // Set the insertion location at the end of the BBEntry.
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
              DL);

          OMPBuilder.createTaskwait(Loc);
        } else if (Dir == OMPD_target) {
          // TODO: remove this special check that comes from the OMP.DEVICE tag
          // and generalize. We only need to generate offloading entries.
          if (IsOpenMPDevice) {
            // Emit the Numba wrapper offloading function.
            SmallVector<Type *, 8> WrapperArgsTypes;
            SmallVector<StringRef, 8> WrapperArgsNames;
            for (auto &It : DSAValueMap) {
              Value *V = It.first;
              DSAType DSA = It.second;

              switch (DSA) {
              case DSA_FIRSTPRIVATE:
                WrapperArgsTypes.push_back(
                    V->getType()->getPointerElementType());
                WrapperArgsNames.push_back(V->getName());
                break;
              case DSA_PRIVATE:
                // do nothing
                break;
              default:
                WrapperArgsTypes.push_back(V->getType());
                WrapperArgsNames.push_back(V->getName());
              }
            }

            Twine DevFuncName = "__omp_offload_numba_" + Fn->getName();
            FunctionType *NumbaWrapperFnTy =
                FunctionType::get(OMPBuilder.Void, WrapperArgsTypes,
                                  /* isVarArg */ false);
            Function *NumbaWrapperFunc = Function::Create(
                NumbaWrapperFnTy, GlobalValue::ExternalLinkage, DevFuncName, M);

            // Name the wrapper arguments for readability.
            for (size_t I = 0; I < NumbaWrapperFunc->arg_size(); ++I)
              NumbaWrapperFunc->getArg(I)->setName(WrapperArgsNames[I]);

            IRBuilder<> Builder(
                BasicBlock::Create(M.getContext(), "entry", NumbaWrapperFunc));
            // Set up default arguments. Depends on the target architecture.
            // TODO: Find a nice way to abstract this.
            FunctionCallee DevFuncCallee(Fn);
            SmallVector<Value *, 8> DevFuncArgs;
            Triple TargetTriple(M.getTargetTriple());
            Value *RetPtr = Builder.CreateAlloca(
                Fn->getArg(0)->getType()->getPointerElementType());
            DevFuncArgs.push_back(RetPtr);
            if (!TargetTriple.isNVPTX()) {
              Value *ExcInfo = Builder.CreateAlloca(
                  Fn->getArg(1)->getType()->getPointerElementType());
              DevFuncArgs.push_back(ExcInfo);
            }

            for (auto &Arg : NumbaWrapperFunc->args())
              DevFuncArgs.push_back(&Arg);

            if (TargetTriple.isNVPTX()) {
              OpenMPIRBuilder::LocationDescription Loc(Builder);
              auto IP = OMPBuilder.createTargetInit(Loc, /* IsSPMD */ false,
                                          /* RequiresFullRuntime */ true);
              Builder.restoreIP(IP);
            }

            Builder.CreateCall(DevFuncCallee, DevFuncArgs);

            if (TargetTriple.isNVPTX()) {
              OpenMPIRBuilder::LocationDescription Loc(Builder);
              OMPBuilder.createTargetDeinit(Loc, /* IsSPMD */ false,
                                            /* RequiresFullRuntime */ true);
            }

            Builder.CreateRetVoid();

            if (TargetTriple.isNVPTX()) {
              constexpr int OMP_TGT_GENERIC_EXEC_MODE = 1;
              // Emit OMP device globals and metadata.
              auto *ExecModeGV =
                  new GlobalVariable(M, OMPBuilder.Int8, /* isConstant */ false,
                                     GlobalValue::WeakAnyLinkage,
                                     Builder.getInt8(OMP_TGT_GENERIC_EXEC_MODE),
                                     DevFuncName + "_exec_mode");
              appendToCompilerUsed(M, {ExecModeGV});

              // Get "nvvm.annotations" metadata node.
              NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

              Metadata *MDVals[] = {ConstantAsMetadata::get(NumbaWrapperFunc),
                                    MDString::get(M.getContext(), "kernel"),
                                    ConstantAsMetadata::get(
                                        ConstantInt::get(OMPBuilder.Int32, 1))};
              // Append metadata to nvvm.annotations.
              MD->addOperand(MDNode::get(M.getContext(), MDVals));

              // Add a function attribute for the kernel.
              Fn->addFnAttr(Attribute::get(M.getContext(), "kernel"));

            } else {
              Constant *OMPOffloadEntry;
              emitOMPOffloadingEntry(M, OMPBuilder, TgtOffloadEntryTy,
                                     DevFuncName, NumbaWrapperFunc,
                                     OMPOffloadEntry);
            }
            continue;
          }

          const DebugLoc DL = BBEntry->getTerminator()->getDebugLoc();
          OpenMPIRBuilder::LocationDescription Loc(
              InsertPointTy(BBEntry, BBEntry->getTerminator()->getIterator()),
              DL);
          Constant *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(Loc);
          Value *SrcLoc = OMPBuilder.getOrCreateIdent(SrcLocStr);

          FunctionCallee TargetMapper = OMPBuilder.getOrCreateRuntimeFunction(
              M, OMPRTL___tgt_target_mapper);
          OMPBuilder.Builder.SetInsertPoint(BBEntry->getTerminator());

          // Emit mappings.
          OffloadingMappingArgsTy OffloadingMappingArgs;
          InsertPointTy AllocaIP(&Fn->getEntryBlock(),
                                 Fn->getEntryBlock().getFirstInsertionPt());
          emitOMPOffloadingMappings(M, OMPBuilder, AllocaIP, DSAValueMap,
                                    StructMappingInfoMap,
                                    OffloadingMappingArgs);

          auto *OffloadResult = OMPBuilder.Builder.CreateCall(
              TargetMapper,
              {SrcLoc, ConstantInt::get(OMPBuilder.Int64, -1),
               ConstantExpr::getBitCast(OMPRegionId, OMPBuilder.Int8Ptr),
               ConstantInt::get(OMPBuilder.Int32, OffloadingMappingArgs.Size),
               OffloadingMappingArgs.BasePtrs, OffloadingMappingArgs.Ptrs,
               OffloadingMappingArgs.Sizes, OffloadingMappingArgs.MapTypes,
               OffloadingMappingArgs.MapNames,
               // TODO: offload_mappers is null for now.
               Constant::getNullValue(OMPBuilder.VoidPtrPtr)});
          auto *Failed = OMPBuilder.Builder.CreateIsNotNull(OffloadResult);
          OMPBuilder.Builder.CreateCondBr(Failed, StartBB, EndBB);
          BBEntry->getTerminator()->eraseFromParent();
        } else {
          LLVM_DEBUG(dbgs() << "Unknown directive " << *CBEntry << "\n");
          assert(false && "Unknown directive");
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
