# deposit_demo.py —— 钱是怎么蒸发的
import redis
import threading
import time

r = redis.Redis(host="127.0.0.1", port=6380, decode_responses=True)
r.set("bank", "0")                      # 公共钱包，清零


def deposit():                          # 一个"存钱员"：加 500 轮，每轮 +50
    for _ in range(500):
        #bal = int(r.get("bank") or 0)   # ① 读：看一眼钱包
                    
        #r.set("bank", str(bal + 50))    # ② 写：把"刚才看到的+50"整个盖回去
        r.incr("bank",50)


t1 = threading.Thread(target=deposit)   # 雇第二个存钱员（线程=第二个人）
t2 = threading.Thread(target=deposit)
t1.start()                              # 俩人同时上岗
t2.start()
t1.join()                               # 我站在门口等：t1 下班我才继续
t2.join()

final = int(r.get("bank"))
print("期望:", 2 * 500 * 50, "实际:", final, "蒸发:", 2 * 500 * 50 - final)