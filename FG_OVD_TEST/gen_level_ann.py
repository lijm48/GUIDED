
import json
from AEsir_utils.data_utils.data_visual import print_json_structure
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import asyncio

init_ann_file = "/data1/liangzhijia/FG-OVD/validation_sets/3_attributes.json"
TOTAL_TASK_CNT, COMPLETED_TASK_CNT = 0, 0

PROMPT = """
use a fine grained vocaborary to generate five simple vocaborary.
You should return a sentence directly, strictly following the following format:
vocab_1 | vocab_2 | vocab_3 | vocab_4 | vocab_5

example:

1.
input: 
A black belt with a metal loop, a metal buckle and a leather strap.
output:
A belt | A black belt | A black belt with a loop | A black belt with a metal loop | A black belt with a metal loop and a metal buckle

2.
input:
A bicycle made of metal with black and white wheels, black handlebar, black and red head tube, red top tube, red down tube and black fork.
output:
A bicycle | A bicycle made of metal | A bicycle made of metal with wheels | A bicycle made of metal with black and white wheels | A bicycle made of metal with black and white wheels and black handlebar

now input:

"""

async def asy_get_simple_vocabs(query_vocab): # 修改为异步函数 async def
    init_prompt = PROMPT

    client = AsyncOpenAI( # 使用 AsyncOpenAI 客户端
        api_key="sk-Jr5d1795a1465d235208fe07658473019ea18ce0c08BLsyK",
        base_url="https://api.gptsapi.net/v1",
    )
    completion = await client.chat.completions.create( # 使用 await 调用异步API
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": init_prompt + query_vocab}
        ]
    )
    all_vocab = completion.choices[0].message.content
    vocab_lst = all_vocab.split('|')

    await client.close() # 显式关闭异步客户端，释放资源 (虽然通常不是严格必需的，但推荐这样做)

    global COMPLETED_TASK_CNT
    COMPLETED_TASK_CNT += 1
    print(f"Completed task {COMPLETED_TASK_CNT}/{TOTAL_TASK_CNT}")

    return vocab_lst






async def asy_ann_opt(): # 修改 ann_opt 为异步函数
    ann_data = json.load(open(init_ann_file, 'r'))

    id_2_category = {item["id"]: item["name"] for item in ann_data["categories"]}

    tasks = [] # 创建一个列表来存储异步任务
    print("create tasks...")
    for i in tqdm(range(len(ann_data["annotations"]))):
        pos_category = id_2_category[ann_data["annotations"][i]["category_id"]]
        task = asy_get_simple_vocabs(pos_category) # 创建异步任务，但不立即执行
        tasks.append(task) # 将任务添加到列表

    global TOTAL_TASK_CNT
    TOTAL_TASK_CNT = len(tasks)

    level_vocab_lsts = await asyncio.gather(*tasks) # 并发执行所有异步任务，并等待全部完成

    for i in range(len(ann_data["annotations"])): # 循环遍历注释，并将结果添加回去
        ann_data["annotations"][i]["level_vocab"] = level_vocab_lsts[i] # 注意这里 level_vocab_lsts 是一个列表的列表

    json.dump(ann_data, open("FG_OVD_TEST/level_3_attributes_val.json", 'w'), indent=4, ensure_ascii=False) # 保存到新的文件，文件名修改为 *_async.json*，添加 indent 和 ensure_ascii 参数方便查看


def show_ann(file):
    with open(file, 'r') as f:
        data = json.load(f)
    id_2_category = {item["id"]: item["name"] for item in data["categories"]}
    # print_json_structure(data)
    # for item in data["annotations"]:
    #     print(item["id"], item["category_id"], id_2_category[item["category_id"]])
    #     print("\n")
    #     print(item["level_vocab"])
    #     print("========================================================================================")
    #     break
    img_id_lst = []
    for item in data["annotations"]:
        img_id_lst.append(item["image_id"])
    
    # get cnt of each img
    img_id_cnt = {}
    for img_id in img_id_lst:
        if img_id not in img_id_cnt:
            img_id_cnt[img_id] = 1
        else:
            img_id_cnt[img_id] += 1

    print(img_id_cnt)

def get_simple_vocabs(query_vocab):

    with open("FG_OVD_TEST/prompt.txt", 'r') as f:
        init_prompt = f.read()

    client = OpenAI(
        api_key="sk-Jr5d1795a1465d235208fe07658473019ea18ce0c08BLsyK",
        base_url="https://api.gptsapi.net/v1",
        )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        
        messages=[
            {"role": "user", "content": init_prompt + query_vocab}
        ]
    )
    all_vocab = completion.choices[0].message.content
    vocab_lst = all_vocab.split('|')
    
    return vocab_lst



def ann_opt():
    ann_data = json.load(open(init_ann_file, 'r'))

    id_2_category = {item["id"]: item["name"] for item in ann_data["categories"]}

    for i in tqdm(range(len(ann_data["annotations"]))):
        pos_category = id_2_category[ann_data["annotations"][i]["category_id"]]
        level_vocab_lst = get_simple_vocabs(pos_category)
        
        ann_data["annotations"][i]["level_vocab"] = level_vocab_lst

    json.dump(ann_data, open("FG_OVD_TEST/level_3_attributes_val.json", 'w'))


if __name__ == "__main__": # 确保在主程序入口处运行异步函数
    # asyncio.run(asy_ann_opt()) # 使用 asyncio.run() 运行异步主函数
    show_ann("dataset/FG_OVD/level_3_attributes_val.json")


