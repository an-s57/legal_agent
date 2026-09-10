import redis, threading, time
r = redis.Redis(host="127.0.0.1", port=6380, decode_responses=True)
r.delete("lock:demo")            # 清场
wins = 0
def grab(name):
    global wins
    got = r.set("lock:demo", name, nx=True,ex=10)   # 抢！
    if got:
        wins += 1
        print(f"{name} 抢到锁")
        raise RuntimeError("模拟崩溃") 
        #time.sleep(0.2)          # 假装干活
       # r.delete("lock:demo")    # 还锁
    else:
        print(f"{name} 抢失败，走了")
ts = [threading.Thread(target=grab, args=(f"T{i}",)) for i in range(10)]
for t in ts: t.start()
for t in ts: t.join()
print("这一轮抢到锁的进程数:", wins, "（期望恰好 1）")