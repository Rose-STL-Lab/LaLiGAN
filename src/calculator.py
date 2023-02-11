# File with basic calculator functions

from functions import run_op
from functions import update_state
from functions import get_operand_count
from functions import get_operator_count


def run_calc():
    
    current_state = {
        'operands': [],
        'operators' : [],
    }

    while True:

        # Update current state by taking input

        current_state = update_state(current_state)

        # Check if we are ready for a computation

        if get_operand_count(current_state) == 2 and get_operator_count(current_state) == 1:
     
            result = run_op(current_state)
            print(current_state['operands'][0], current_state['operators'][0], current_state['operands'][1], '=', result)
     
            current_state = {
                'operands': [],
                'operators' : [],
            }


if __name__ == "__main__":
    run_calc()
