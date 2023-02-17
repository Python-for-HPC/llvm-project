#ifndef LLVM_TRANSFORMS_INTRINSICS_OPENMP_CODEGEN_H
#define LLVM_TRANSFORMS_INTRINSICS_OPENMP_CODEGEN_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

using namespace llvm;
using namespace omp;

using InsertPointTy = OpenMPIRBuilder::InsertPointTy;
using BodyGenCallbackTy = OpenMPIRBuilder::BodyGenCallbackTy;
using FinalizeCallbackTy = OpenMPIRBuilder::FinalizeCallbackTy;

namespace iomp {
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
    {"DIR.OMP.TARGET", OMPD_target},
    {"DIR.OMP.TEAMS", OMPD_teams},
    {"DIR.OMP.TARGET.TEAMS", OMPD_target_teams},
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
    {"QUAL.OMP.MAP.TOFROM.STRUCT", DSA_MAP_TOFROM_STRUCT}};

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

class CGIntrinsicsOpenMP {
public:
  CGIntrinsicsOpenMP(Module &M);

  OpenMPIRBuilder OMPBuilder;
  Module &M;
  StructType *TgtOffloadEntryTy;

  StructType *getTgtOffloadEntryTy() { return TgtOffloadEntryTy; }

  void emitOMPParallel(MapVector<Value *, DSAType> &DSAValueMap,
                       const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry,
                       BasicBlock *StartBB, BasicBlock *EndBB,
                       BasicBlock *AfterBB, FinalizeCallbackTy FiniCB,
                       Value *IfCondition, Value *NumThreads);

  void emitOMPParallelDevice(MapVector<Value *, DSAType> &DSAValueMap,
                             const DebugLoc &DL, Function *Fn,
                             BasicBlock *BBEntry, BasicBlock *StartBB,
                             BasicBlock *EndBB, BasicBlock *AfterBB,
                             FinalizeCallbackTy FiniCB, Value *IfCondition,
                             Value *NumThreads);

  void emitOMPFor(MapVector<Value *, DSAType> &DSAValueMap, Value *IV,
                  Value *UB, BasicBlock *PreHeader, BasicBlock *Exit,
                  OMPScheduleType Sched, Value *Chunk, bool IsStandalone);

  void emitOMPTask(MapVector<Value *, DSAType> &DSAValueMap, Function *Fn,
                   BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
                   BasicBlock *AfterBB);

  void emitOMPOffloadingEntry(const Twine &DevFuncName, Value *EntryPtr,
                              Constant *&OMPOffloadEntry);

  void
  emitOMPOffloadingMappings(InsertPointTy AllocaIP,
                            MapVector<Value *, DSAType> &DSAValueMap,
                            MapVector<Value *, SmallVector<FieldMappingInfo, 4>>
                                &StructMappingInfoMap,
                            OffloadingMappingArgsTy &OffloadingMappingArgs);

  void emitOMPSingle(Function *Fn, BasicBlock *BBEntry, BasicBlock *AfterBB,
                     BodyGenCallbackTy BodyGenCB, FinalizeCallbackTy FiniCB);

  void emitOMPCritical(Function *Fn, BasicBlock *BBEntry, BasicBlock *AfterBB,
                       BodyGenCallbackTy BodyGenCB, FinalizeCallbackTy FiniCB);

  void emitOMPBarrier(Function *Fn, BasicBlock *BBEntry, Directive DK);

  void emitOMPTaskwait(BasicBlock *BBEntry);

  void emitOMPTarget(StringRef DevFuncName, ConstantDataArray *ELF,
                     Function *Fn, BasicBlock *BBEntry, BasicBlock *StartBB,
                     BasicBlock *EndBB,
                     MapVector<Value *, DSAType> &DSAValueMap,
                     MapVector<Value *, SmallVector<FieldMappingInfo, 4>>
                         &StructMappingInfoMap,
                     Value *NumTeams, Value *ThreadLimit);

  void emitOMPTargetDevice(Function *Fn,
                           MapVector<Value *, DSAType> &DSAValueMap);

  void emitOMPTeams(MapVector<Value *, DSAType> &DSAValueMap,
                    const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry,
                    BasicBlock *StartBB, BasicBlock *EndBB, BasicBlock *AfterBB,
                    Value *NumTeams, Value *ThreadLimit);

  void emitOMPTeamsDevice(MapVector<Value *, DSAType> &DSAValueMap,
                          const DebugLoc &DL, Function *Fn, BasicBlock *BBEntry,
                          BasicBlock *StartBB, BasicBlock *EndBB,
                          BasicBlock *AfterBB);

  GlobalVariable *emitOffloadingGlobals(StringRef DevWrapperFuncName,
                                        ConstantDataArray *ELF);

  Twine getDevWrapperFuncPrefix() { return "__omp_offload_numba_"; }

  Function *createOutlinedFunction(
      MapVector<Value *, DSAType> &DSAValueMap, Function *OuterFn,
      BasicBlock *BBEntry, BasicBlock *StartBB, BasicBlock *EndBB,
      BasicBlock *AfterBB, SmallVectorImpl<llvm::Value *> &CapturedVars);
};

} // namespace iomp

#endif