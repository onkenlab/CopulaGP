with open("list.sh","w") as f:
    exp = 'ST260_Day1_ygl'
    for i in range(20):
        f.write(f"python train.py  -exp {exp} -g -start {i} -layers {i+1}\n")
