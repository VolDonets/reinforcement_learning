from blackjack_hidden_deck_env import BlackjackHiddenDeckEnv


action_map = {
    1: 'Hit',
    0: 'Stick'
}
seed = 2
episodes = 10_000

hidden_deck = {
    1: 10,
    2: 30,
    3: 30,
    4: 20,
    5: 10,
    6: 5,
    7: 3,
    8: 1,
    9: 1,
    10: 1,
    10: 1,
    10: 1,
    10: 1
}
env = BlackjackHiddenDeckEnv(hidden_deck)
env.seed(seed)

wins = 0
loss = 0
draws = 0

for e in range(1, episodes+1):
    state = env.reset()
    print(f'===== Episode: {e} =====')
    while True:
        agent_sum = state[0]
        dealer_sum = state[1]

        if agent_sum < 17:
            action = 1
        else:
            if agent_sum < dealer_sum:
                action = 1
            else:
                action = 0

        next_state, reward, done, _ = env.step(action)
        print(f'state {state} | action: {action_map[action]} '
              f'| reward: {reward} | next state: {next_state}')

        if done:
            if reward > 0:
                wins += 1
            elif reward < 0:
                loss += 1
            else:
                draws += 1
            break

        state = next_state

print(f'Wins: {wins} | Loss: {loss} | Draw: {draws}')
