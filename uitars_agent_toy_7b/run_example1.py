
"""

env
agent
 run task

"""
import os,json
import logging
import datetime
import time

from uitars_agent_428 import UITARSAgent
from env_uos import Env
import argparse

def run_1example(instruction,maxStep=10,name='1',logging=None,env=None,agent=None,example_result_dir=None):
    logging.info('\n'+instruction+'\n...........\n')
    agent.reset()
    env.reset()

    max_steps=maxStep
    obs = env._get_obs() # Get the initial observation
    done = False
    wait_for_user=False
    #instruction='最小化当前应用'
    step_idx = -1

    ret_stats='fail'



    while not done and not wait_for_user and step_idx < max_steps:
        logging.info(obs)
        tmp= agent.predict(
            instruction,
            obs
        )
        response, actions=tmp
        logging.info({'response':response,'actions':actions})
        step_idx+=1
        for action in actions:
            print(action)
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logging.info("Step %d: %s", step_idx , action )
            if 'DONE' in action:
                done=True
                ret_stats='done'
            elif 'wait' in action:
                ret_stats='wait for user'
                wait_for_user=True

            else:
                obs, reward, done, info = env.step(action  )
            ##logger.info("Reward: %.2f", reward)
            #logger.info("Done: %s", done)
            # Save screenshot and trajectory information
            # with open(os.path.join(example_result_dir, f"step_{step_idx  }_{action_timestamp}.png"),
            #           "wb") as _f:
            #     _f.write(obs['screenshot'])
            #curr_img1=os.path.join(example_result_dir, f"step_{step_idx  }_{action_timestamp}.png")

            with open(os.path.join(example_result_dir, f"{name}_trajectory.jsonl"), "a") as f:
                f.write(json.dumps({
                    "instruction":instruction,
                    "step_num": step_idx  ,
                    "action_timestamp": action_timestamp,
                    "action": actions,
                    'response':response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file":obs['screenshot'],
                    'return_state':ret_stats
                    #"screenshot_file": f"step_{step_idx }_{action_timestamp}.png"
                },indent=4,ensure_ascii=False))
                f.write("\n")
            if done:
                logging.info("The episode is done.")
                break
            if wait_for_user:
                logging.info("The episode is done for calling user")
    return ret_stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--example_result_dir', default='./tmp1', type=str,
                            help='log path, screenshot save path,')
    parser.add_argument('-s', '--max_step', default=3, type=int,
                        help='max step')
    parser.add_argument('-t', '--task', default="打开当前编辑器应用的设置菜单，设置快捷键", type=str,
                        help='task or instruction')



    return parser.parse_args()


if __name__=='__main__':

    args = get_args()
    from request1 import send_fake_req_7b,send_post_action
    from parse_action_try import parsing_response_to_pyautogui_code, parse_action_output

    parse_fn = parse_action_output

    agent = UITARSAgent( num_history_use_num=6,
        req_fn=send_post_action,##请求
                         parse_fn=parse_fn,### 解析模型的输出
                         action2pyautogui_fn=parsing_response_to_pyautogui_code### 根据每个actiontype,attribute 分别写pyautogui
                         )##  依赖模型的  请求 ，解析,写pyautogui

    #### env用于存储截屏 的路径，以及Log
    example_result_dir =args.example_result_dir
    if not  os.path.exists(example_result_dir):
        os.makedirs(example_result_dir)
    env = Env(screenshot_dir=example_result_dir)
    # Configure logging
    logging.basicConfig(filename=os.path.join(example_result_dir,'env_log.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    instr=args.task
    print(instr)

    maxstep=args.max_step # 单步的调成1
    outname=str(time.time())
    return_state=run_1example(instr,
                 maxstep,
                 outname,
                 logging,
                 env,
                 agent,
                 example_result_dir)
    ###
    #########
    # 经过若干步骤后 终止trajectory的原因有3 ：
    # 1 wait calling for user
    # 2  done
    # 3 not done,fail  超过最大步数 还没done 也没有wait for calling user