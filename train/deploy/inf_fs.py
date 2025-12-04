import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run legal document generation with specified model suffix.')
    parser.add_argument('--suffix', type=str, required=True, help='Suffix of the model path')
    parser.add_argument('--dataset_path', type=str, default="../data/test.json", help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default="../output/fewshot/output.json", help='Path to save the output')
    return parser.parse_args()

# 如果输出包含 </think>，仅保留其后的正文
def strip_think(text: str) -> str:
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text

# Function to generate reasoning given a question
def generate_reasoning(fact):
    input_content = f"""
案件事实：{fact}
要求：请根据上面提供的事实描述，生成一篇完整且具有法律效力的中文的刑事判决书。生成的文书必须结构严谨、逻辑清晰，请严格按下列模板输出，不得增删字段：
1. 开头
2. 事实描述 :请直接复述提供的事实描述，不得进行删减或改动
3. 司法理由 : 根据上述事实描述，结合相关刑法条款、法律原则和司法解释，详细论述案件的法律分析，以“本院认为”开头。  分析内容应包括：对证据的评估  ,犯罪构成要件的论证  ,相关法律条文的引用及其适用说明  ,请确保推理过程严谨、论证充分，为判决结果提供充分法律依据。
4. 判决结果 :在此部分明确给出法院的最终判决，以“判决如下”开头。  判决内容应具体包括处罚措施（如刑期、罚金、附加刑等）及其法律依据，确保与前述司法理由相呼应，文书整体逻辑连贯。
注意：确保文书所有部分均符合真实司法文书的写作规范，语言应正式、客观、清晰。
示例：
刑法判决书:  
许帅伟妨害公务一审刑事判决书 河南省登封市人民法院 刑事判决书 （2017）豫0185刑初1244号 公诉机关河南省登封市人民检察院。 被告人许帅伟，男，1979年8月24日出生于河南省登封市，汉族，中专文化程度，中共党员，郑州市嵩阳煤机制造有限公司员工，住登封市。因涉嫌犯妨害公务罪于2017年8月15日被登封市公安局刑事拘留，同年8月22日被取保候审。 指定辩护人景俊娜，河南群达律师事务所律师。 登封市人民检察院以登检刑诉［2017］616号起诉书指控被告人许帅伟犯妨害公务罪，于2017年11月17日向本院提起公诉，并建议本院适用速裁程序审理。依据《全国人民代表大会常务委员会关于授权最高人民法院、最高人民检察院在部分地区开展刑事案件认罪认罚从宽制度试点工作的决定》，本院决定适用速裁程序，实行独任审判，公开开庭审理了本案。登封市人民检察院检察员梁书伟出庭支持公诉。被告人许帅伟及其指定辩护人均到庭参加了诉讼。现已审理终结。 登封市人民检察院指控，2017年8月14日17时许，被告人 许帅伟在登封市少林办事处耿庄登封市第二游泳馆游泳期间，与他人发生争执并打架，登封市公安局少林派出所民警乔登辉带领辅警曹某和常某现场处警，期间，被告人许帅伟抗拒民警执法，并拿起杯具砸伤辅警曹某。经鉴定，曹某面部的损伤程度构成轻微伤。 为证明该指控事实，公诉机关向本院提供了被告人许帅伟的供述；证人乔登辉、常某、曹某、裴某等人的证言；辨认笔录；现场勘验检查笔录及照片；登封市公安局处警表；鉴定书；视听资料；血醇鉴定意见书；收到条、撤诉书；公安机关出具的户籍证明、无前科证明及到案经过等证据。据此认为被告人许帅伟的行为已构成妨害公务罪，并建议本院对被告人许帅伟判处拘役三至五个月。 经法庭审理，查明的事实、证据与指控的事实、证据相同，被告人许帅伟及其指定辩护人对起诉书指控的事实、罪名、证据及量刑建议均不持异议，且被告人签字具结。本院予以确认。 本院认为，被告人许帅伟以暴力方法阻碍公安机关依法执行公务，其行为已构成妨害公务罪。公诉机关指控被告人许帅伟犯妨害公务罪的事实、罪名及理由成立，量刑建议适当，依法予以支持。被告人许帅伟归案后，如实供述自己的罪行，可以从轻处罚；其自愿认罪，已赔偿受伤辅警经济损失，可酌情从轻处罚。与之相对应的辩护意见成立，本院予以采纳。 根据《中华人民共和国刑法》第二百七十七条第一款和第五款、第六十七条第三款之规定，判决如下： 被告人许帅伟犯妨害公务罪，判处拘役三个月。（刑期自判决执行之日起计算。判决执行前先行羁押的，羁押一日折抵刑期一日。即自2017年11月17日起至2018年2月8日止。原被羁押的8日已予折抵。） 如不服本判决，可在接到判决书的第二日起十日内，通过本院或直接向河南省郑州市中级人民法院提出上诉。书面上诉的，应当提交上诉状正本一份，副本十五份。 审判员杨海洋 二〇一七年十一月十七日 书记员张宇

请参考以上示例，根据案件事实生成一份刑事判决书，结构完整，严格按照要求格式。

"""
    messages = [
        {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": input_content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return strip_think(response)



# Function to process a dataset
def process_dataset(dataset_path, output_path):
    # Load the dataset
    test_result = []
    with open(dataset_path, "r") as f:
        for line in tqdm(f):
            one = json.loads(line.strip())
            fact = one.get("input") or one.get("text") or one.get("Fact")
            exp_ans = one.get("output") or one.get("fd")
            gen_ans = generate_reasoning(fact)
            print(f"Generated answer: {gen_ans}")

            # Update data and add to results list
            entry = {
                "gen_ans": gen_ans,
                "exp_ans": exp_ans
            }
            test_result.append(entry)

    # Save the results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")

def main():
    args = parse_arguments()
    model_name = args.suffix

    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    process_dataset(args.dataset_path, args.output_path)

if __name__ == "__main__":
    main()
