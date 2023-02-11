import sys

from src.functions import run_op

class TestOps:
    
    def test_add(self):
        current_state = {
            'operands' : [2, 3],
            'operators': ['+']
        }

        result = run_op(current_state)
        assert(result == 5.0)
