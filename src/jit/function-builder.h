/*
 * Copyright 2017 wasmjit-omr project participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUNCTIONBUILDER_HPP
#define FUNCTIONBUILDER_HPP

#include "type-dictionary.h"
#include "stack.h"
#include "ilgen/BytecodeBuilder.hpp"
#include "ilgen/MethodBuilder.hpp"
#include "ilgen/VirtualMachineOperandStack.hpp"

#include "src/interp.h"

namespace wabt {
namespace jit {

class FunctionBuilder : public TR::MethodBuilder {
 public:
  FunctionBuilder(interp::Thread* thread, interp::DefinedFunc* fn, TypeDictionary* types);
  bool buildIL() override;

  /**
   * @brief Generate push to the interpreter stack
   * @param b is the builder object used to generate the code
   * @param type is the name of the field in the Value union corresponding to the type of the value being pushed
   * @param value is the IlValue representing the value being pushed
   */
  void Push(TR::IlBuilder* b, const char* type, TR::IlValue* value);
  void Push(TR::IlBuilder* b, TR::IlValue* value) {
    Push(b, TypeFieldName(value->getDataType()), value);
  }

  /**
   * @brief Generate pop from the interpreter stack
   * @param b is the builder object used to generate the code
   * @param type is the name of the field in the Value union corresponding to the type of the value being popped
   * @return an IlValue representing the popped value
   */
  TR::IlValue* Pop(TR::IlBuilder* b, const char* type);

  /**
   * @brief Drop a number of values from the interpreter stack, optionally keeping the top value of the stack
   * @param b is the builder object used to generate the code
   * @param drop_count is the number of values to drop from the stack
   * @param keep_count is 1 to keep the top value intact and 0 otherwise
   */
  void DropKeep(TR::IlBuilder* b, uint32_t drop_count, uint8_t keep_count);

  /**
   * @brief Generate load of pointer to a vlue on the interpreter stack by an index
   * @param b is the builder object used to generate the code
   * @param depth is the index from the top of the stack
   * @return and IlValue representing a pointer to the value on the stack
   *
   * JitBuilder does not currently represent unions as value types. This a problem
   * for this function because it cannot simply return an IlValue representing
   * the union. As workaround, it will generate a load of the *base address* of
   * the union, instead of loading the union directly. This behaviour differs
   * from `Thread::Pick()` and users must take this into account.
   */
  TR::IlValue* Pick(TR::IlBuilder* b, Index depth);

 private:
  struct BytecodeWorkItem {
    TR::BytecodeBuilder* builder;
    VirtualStack stack;
    const uint8_t* pc;

    BytecodeWorkItem(TR::BytecodeBuilder* builder,
                     VirtualStack stack,
                     const uint8_t* pc)
      : builder(builder), stack(stack), pc(pc) {}
  };

  void SetUpLocals(const uint8_t** pc, VirtualStack* stack);
  uint32_t GetLocalOffset(VirtualStack* stack, Type* type, uint32_t depth);

  template <typename T>
  const char* TypeFieldName() const;
  const char* TypeFieldName(Type t) const;
  const char* TypeFieldName(TR::DataType dt) const;

  template <typename T, typename TOpHandler>
  void EmitBinaryOp(VirtualStack* stack, TOpHandler h);

  template <typename T, typename TOpHandler>
  void EmitUnaryOp(VirtualStack* stack, TOpHandler h);

  template <typename T>
  void EmitIntDivide(TR::IlBuilder* b, VirtualStack* stack);

  template <typename T>
  void EmitIntRemainder(TR::IlBuilder* b, VirtualStack* stack);

  template <typename>
  TR::IlValue* CalculateShiftAmount(TR::IlBuilder* b, TR::IlValue* amount);

  std::vector<BytecodeWorkItem> workItems_;

  interp::Thread* thread_;
  interp::DefinedFunc* fn_;

  TR::IlType* const valueType_;
  TR::IlType* const pValueType_;

  bool Emit(TR::BytecodeBuilder* b, VirtualStack* stack, const uint8_t* istream, const uint8_t* pc);
};

}
}

#endif // FUNCTIONBUILDER_HPP
