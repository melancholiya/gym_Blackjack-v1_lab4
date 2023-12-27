
def select_optimal_action(q_table, state, action_space):
    max_q_value_action = None
    max_q_value = float('-inf')
    if q_table[state]:
        for action, action_q_value in q_table[state].items():
            if action_q_value >= max_q_value:
                max_q_value = action_q_value
                max_q_value_action = action
    return max_q_value_action if max_q_value_action is not None else action_space.sample()