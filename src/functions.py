def get_operand_count(current_state):

    return len(current_state.get('operands'))


def get_operator_count(current_state):

    return len(current_state.get('operators'))


def update_state(current_state):
    
    if get_operand_count(current_state) == 0:

        val = input('Input operand1 : ')
        current_state['operands'].append(float(val))

    elif get_operand_count(current_state) == 1 and get_operator_count(current_state) == 0:

        val = input('Input operator : ')
        current_state['operators'].append(val)

    elif get_operand_count(current_state) == 1 and get_operator_count(current_state) == 1:
        val = input('Input operand2 : ')
        current_state['operands'].append(float(val))

    return current_state


def run_op(current_state):

    op1 = current_state.get('operands')[0]
    op2 = current_state.get('operands')[1]

    op = current_state.get('operators')[0]

    if op == '+':
        return op1 + op2
    elif op == '-':
        return op1 - op2
    elif op == '*':
        return op1 * op2
    elif op == '/':
        return op1 / op2
