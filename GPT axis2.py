from openai import OpenAI

client = OpenAI(api_key="")


import pandas as pd
import os


dataset = pd.read_csv("input.csv")


def analyze_conversation(previous_turn, current_turn, response):
    prompt = f"""
   Analyze the conversation flow and categorize it based on the following taxonomies:
   These are meant to categorize the method of the follow-up query Q2 by taking into account the previous turn Q1.

        1) 조건 제외
        Def: 사용자가 이전 질의에 대해 특정 조건을 명시적으로 제거함
        EG:
        Q1: 유후인으로 부모님 모시고 가려고 정했는데 코스 좀 짜줘
        Q2: 유후인 코스 좀 짜줘

        Q1: 숙취에 도움되는 이부프로펜 성분의 진통제를 추천해줘
        Q2: 숙취에 도움되는 진통제를 추천해줘

        2) 조건 추가 및 명시 This can be used simultaneously with other themes.
        Def: 사용자가 이전 질의에 대해 추가적인 조건이나 매개변수를 명시하여 포함시킴
        EG:
        Q1: 신혼부부 집들이 선물 골라줘
        Q2: 5만원 대의 신혼부부 집들이 선물 골라줘

        Q1: 허리에 좋은 운동 뭐가 있을까
        Q2: 허리디스크를 가진 사람이 하기에 좋은 허리운동은 뭐야?
        
        Q1: 신혼부부들이 제일 선호하는 식기세척기 종류는 뭐야
        Q2: 신혼부부들이 제일 선호하는 식기세척기 제품명 알려줘

        Q1: 러시아 쿠데타에 대해 알려줘
        Q2: 최근에 일어난 바그너 그룹 관련된 쿠데타에 대해서 더 자세히 알려줘

        Q1: 카이스트에서 제일 가까운 스타벅스 위치 알려줘
        Q2: 대전 카이스트에서 제일 가까운 스타벅스 위치 알려줘

        Q1: 여름에 부산에 여행가려고 하는데
        Q2: 여름에 부산 가면 놀거리

        Q1: 부산 국밥 유명한 곳
        Q2: 부산 주민들의 국밥 추천

        Q1: 앞서 설명한 방법 중 집에서 할 수 있는 방법과 과정 설명해줘
        Q2: 허리에 좋은 운동 중 집에서 할 수 있는 방법을 골라서 과정 설명해줘
        
        3) 조건 치환
        Def: 사용자가 같은 질의 의도를 다른 조건이나 형태로 표현하여 질의함
        EG: 
        Q1: 한국과학기술원에서 제일 가까운 스타벅스 위치 알려줘
        Q2: 어은동에서 제일 가까운 스타벅스 위치 알려줘]

        Q1:  더 비싸고 좋은 커피머신을 알려줘. 그리고 캡슐 커피 머신 종류는 제외해서 알려줘
        Q2: 캡슐 커피 종류를 제외한 비싼 커피 머신 기계를 추천해줘
        Explanation: Q1 and Q2 have the same meaning so 재질의

        Q1: 연마제가 잘 닦였는지 어떻게 확인할 수 있어?
        Q2: 스테인리스 텀블러의 연마제가 잘 제거되었는지 확인하는 방법 알려줘
        Explanation 연마제 has basically been replaced as 스테인리스 연마제 therefore it is 재질의 with alittle bit of variation.

        4) 답변 형식 변환 
        Def:사용자가 답변을 특정 형식(예: 표, 지도, 그래프)으로 제공받기를 원하는 경우.
        EG: 
        Q1: 대전 유성구 어은로 스타벅스에 가는 방법을 물어볼게
        Q2: 유성구 어은 스타벅스를 지도에서 위치를 표시해줄 수 있니

        Q1: 허리에 좋은 운동 뭐가 있을까?
        Q2: 사진으로도 보여줘

        Q1: 숙취해소제 아이솔솔에 대해 알려줘
        Q2: 아이솔솔의 최저가와 링크를 알려줘
        Explanation: Any other type that the user is asking for including link should be 답변 형식 변환

        5) 답변에 대한 반박, 지적 (Criticizing a Response)
        Def: 사용자가 제공된 답변이 원래의 질문과 관련 없거나 적절하지 않다고 지적하는 경우.
        EG: 
        Q1: 수피아가 순 우리말이야?
        Q2: 그런데 왜 나한테 저게 순우리말이라고 했어?

        Q1: 5천원 이하 집들이 선물 알려줘
        Q2: 만원이잖아 오천원 이하라니깐

        6) 답변에 대한 동의, 긍정적 반응
        Def: 사용자가 제공된 답변에 대한 만족을 표현하는 경우
        Q1: 그래 그럼 50만원 이하의 식기세척기라도 찾아보자
        R: 50만원 이하의 식기세척기 제품들을 검색해보았습니다. 
        Q2: 너무 저렴한 것만 찾아오긴 했는데, 그래도 이번엔 틀린 답은 아니네 잘했어 진작 이러지 그랬니?

        7) 답변에 대한 추가 정보 요청 (Requesting Additional Information)
        Def: 사용자가 바로 직전에 검색 엔진으로 제공받은 답변에 대한 추가적인 정보를 요청하는 경우. This taxonomy can be used simultaneously with other themes. If you feel like the Q2 is asking for additional information from R, include it in the possible sets.

        Q1: 위스키는 어때
        R: 5만원 대의 신혼부부 집들이 선물로 추천드릴만한 위스키 제품입니다. 1. 오레포스 Orrefors 위스키 신혼부부 집들이 선물 글래스 시티 하이볼 x4 글래스 시티 하이볼 4개 세트로 구성된 제품입니다. 하이볼은 칵테일용 위스키로, 신혼부부의 집에서도 쉽게 즐길 수 있습니다. 오레포스 브랜드의 위스키로, 고급스러운 디자인과 품질로 인기가 있습니다. 3. 다온 짐빔 버번위스키 온더락잔 아메리칸위스키 신혼부부 집들이선물 아메리칸 위스키 온더락잔 1개 세트로 구성된 제품입니다.
        Q2: 온더락이 뭐야
        Explanation: 온더락 is mentioned directly in the response and the user wants to know what it is. Therefore it is requesting for additional information.

        Q1: 세제 추천해줘
        R: 스테인레스 텀블러 세척에 좋은 세제를 추천해드리겠습니다. 1. 스테인레스 텀블러 세척 편리 보온 보냉 머그컵 스테인레스 텀블러를 세척할 수 있는 편리한 머그컵입니다. 보온, 보냉 기능이 있어 실용적이며, 내열성이 뛰어나 뜨거운 음료나 차가운 음료를 담을 수 있습니다. 텀블러 내부를 깨끗하게 세척할 수 있는 전용 세제가 함께 제공됩니다
        Q2: 세척도구 추천해줘
        Explanation: Q2 세척도구 has very similar correlation to what the response has been saying about 세척 there fore the user is asking additional information from the response. 

        8) 도메인 내 다른 질문  If Q2 is asking an exploratory question within the same domain, include it in the possible sets.
        Def: 동일한 도메인 내에서 서로 다른 세부 주제나 관점에 대해 질문하는 경우.
        EG: 
        Q1: 심한 숙취로 모르고 두통약을 먹어버렸으면 어떻게 해?
        Q2: 과음으로 인한 두통에 좋은건 뭐야?

        Q1: 비스포크 식기세척기 금액은 얼마야
        R: 비스포크 식기세척기 가격 정보입니다. 1. 삼성전자 비스포크 DW60A8355FG 가격: 837640원 부가기능: LED라이트, LED램
        Q2: 제일 선호하는 식기세척기의 용량은?"

        Q1: 그럼 뭘 먹는게 좋아?
        Q2: 아세트아미노펜 성분이 숙취에 왜 안좋은지 알려줘
    
        9) 확인 질문
        Def: 사용자가 이전에 제공된 답변이나 정보에 대해 직접적인 확인이나 재확인을 요청
        EG: 
        Q1: 숙취에 두통약 먹어도 돼?
        Q2: 숙취로 인한 두통에 먹을 수 있는 약은 없어?
        
        Q1: 스테인리스 텀블러 연마제 어떻게 제거하지?
        R: 스테인리스 텀블러의 연마제를 제거하는 방법은 다음과 같습니다. 1. 식용유를 사용한 닦기 - 스테인리스 텀
        Q2: 저번에 1번 방법으로 다이소에서 산 냄비를 세척한 적이 있었는데, 딱히 검은 때가 묻어나오지는 않더라고,,, 1번 방법 효과적인 거 맞아?"

        10) 의견 구하기 This very ambiguous, if you feel like the Q2 is asking for suggestions, include it in the possible sets.
        Def: 사용자가 직전에 검색 엔진으로부터 제공받지 않은 대안이나 선택지를 제시해 추가적인 정보나 의견을 요청하는 경우.
        EG: 
        Q1:  5만원 대의 신혼부부 집들이 선물 골라줘
        R: 집들이선물 신혼부부선물 핸드메이드도자기 소주잔 별헤는 밤 행성술잔 1p 도자기/세라믹 재질로 만들어진 별헤는 밤 행성술잔입니다. 행성 모양과 별들이 골드로 고급스럽게 수놓아져 있으며, 개별 낱개포장도 꼼꼼하게 되어있습니다. 사용자 후기에서는 배송이 빠르고 디자인도 예쁘다는 평가가 많습니다.
        Q2: 위스키는 어때
        Explanation: Although the Response has nothing to do with 위스키, the user is asking for 의견

        Q1: 스테인레스 텀블러 연마제 제거방법
        R: 1. 연마제 제거 방법 - 연마제 제거는 스테인레스 텀블러 세척의 마지막 단계입니다. 연마제 제거를 하지 않으면 스테인리스 텀블러에 남아있는 연마제가 몸에 해로울 수 있습니다. 일반적으로 연마제 제거 방법으로는 식용유나 베이킹 소다
        Q2: 세제 추천해줘
        Explanation, Even though the response has to do with 세제, the user is still asking for opinions as to which 세제 to use. Therefore it is 의견구하기

        11) 무목적 질의 및 대화	
        Def: 명확한 정보 인출이 아닌 대화를 시도하는 경우.
        EG: 
        Q1: 요가나 필라테스중에 더 도움이 되는 쪽이 뭘까
        Q2: 요가를 해본적 있어?

   12) 분류 불가	
   Def: 위에 분류와 전혀 맞지 않는 경우

   Based on these definitions and examples, categorize the following conversation into the specific subcategory that is all labeled with a number:

   Q1: {previous_turn}
   R: {response}
   Q2: {current_turn}

    Only give the output as the possible number or numbers that correlates with the subcategories(1~12) You can give at most three numbers if it fits more than one. separate the possible numbers with comma (EX: 1,2,3). Dont break the format as it is crucial because we are using api to get the data automatically.
  
   """

    try:
        response = client.chat.completions.create(
            model=""//desired model, we utilized most recent gpt 4 turbo model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": previous_turn},
                {"role": "assistant", "content": current_turn},
            ],
        )
        print("Response from OpenAI:")

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error"


for index, row in dataset.iterrows():
    # Find the first turn number for this conversation
    first_turn_no = dataset.loc[
        dataset["conversation_id"] == row["conversation_id"], "turn_no"
    ].min()

    # Check if the current turn is not the first turn
    if row["turn_no"] > first_turn_no:
        try:
            # Getting the previous turn
            previous_turn = dataset.loc[
                (dataset["conversation_id"] == row["conversation_id"])
                & (dataset["turn_no"] == row["turn_no"] - 1),
                "contextual_question",
            ].iloc[0]

            # Getting the response from the previous turn
            response = dataset.loc[
                (dataset["conversation_id"] == row["conversation_id"])
                & (dataset["turn_no"] == row["turn_no"] - 1),
                "response_text",
            ].iloc[0]

            current_turn = row["contextual_question"]
            print(f"\nconversation_id: {row['conversation_id']}")
            print(f"Q1: {previous_turn}")
            # print(f"R: {response}")
            print(f"Q2: {current_turn}")

            taxonomy_number = analyze_conversation(
                previous_turn, current_turn, response
            )
            print(f"Taxonomy Number: {taxonomy_number}")

            # Update the current row's taxonomy based on the analysis
            dataset.at[index, "taxonomy"] = taxonomy_number
        except IndexError:
            # Handle cases where the previous turn is not found
            print(f"Previous turn not found for index {index}")


# Saving the modified dataset
dataset.to_csv("output.csv", index=False)
