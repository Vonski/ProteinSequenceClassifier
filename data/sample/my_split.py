import random
import os

if __name__ == "__main__":
    with open('./human_train_vlen.txt') as f:
        humans_tr = f.read().splitlines()
    with open('./human_val_vlen.txt') as f:
        humans_v = f.read().splitlines()
    with open('./human_test_vlen.txt') as f:
        humans_te = f.read().splitlines()
    humans = humans_tr + humans_v + humans_te

    with open('./mouse_test_vlen.txt') as f:
        mice = f.read().splitlines()

    random.shuffle(humans)
    random.shuffle(mice)

    humans_s1_p = int(len(humans) * 0.8)
    humans_s2_p = int(len(humans) * 0.9)
    mice_s1_p = int(len(mice) * 0.8)
    mice_s2_p = int(len(mice) * 0.9)

    humans_train = humans[:humans_s1_p]
    humans_val = humans[humans_s1_p:humans_s2_p]
    humans_test = humans[humans_s2_p:]
    mice_train = mice[:mice_s1_p]
    mice_val = mice[mice_s1_p:mice_s2_p]
    mice_test = mice[mice_s2_p:]

    os.makedirs('./my_data', exist_ok=True)
    with open('./my_data/humans_train.txt', 'w') as f:
        f.write("\n".join(humans_train))
    with open('./my_data/humans_val.txt', 'w') as f:
        f.write("\n".join(humans_val))
    with open('./my_data/humans_test.txt', 'w') as f:
        f.write("\n".join(humans_test))
    with open('./my_data/mice_train.txt', 'w') as f:
        f.write("\n".join(mice_train))
    with open('./my_data/mice_val.txt', 'w') as f:
        f.write("\n".join(mice_val))
    with open('./my_data/mice_test.txt', 'w') as f:
        f.write("\n".join(mice_test))
