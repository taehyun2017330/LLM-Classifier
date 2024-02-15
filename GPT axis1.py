from openai import OpenAI

client = OpenAI(api_key="")


import pandas as pd
import os


dataset = pd.read_csv("filtered_conversations2_50.csv")


def analyze_conversation(initial_intent, previous_turn, current_turn, response):
    prompt = f"""
   Analyze the conversation flow and categorize it based on the following taxonomies:
   These are meant to categorize the purpose of the follow-up query Q2 by taking into account the previous turn Q1 and the response R.
   I will also provide negative examples which are common examples of mistakes of classifying it as a certain section.

    1) 질의 명확화 (Clarification)
    Def: 사용자가 동일한 의도를 가지고, Follow-up query를 통해 본인의 질의 의도를 명확하게 하는 경우. This is when you feel like Q2 isnt adding or asking anything new but rather a reptition of Q1.
    EG:

    Q1: 2023년 6월 24일에 일어난 러시아 쿠데타에 대해 더 자세히 알려줘
    R:...
    Q2: 2023년 6월 24일, 러시아에서는 '바그너 그룹'을 필두로 군사 반란에 대해 자세히 알려줘 

    Q1: 유후인으로 부모님 모시고 가려고 정했는데 코스 좀 짜줘
    Q2: 유후인 코스 좀 짜줘

    Q1: 연마제가 잘 닦였는지 어떻게 확인할 수 있어?
    Q2: 스테인리스 텀블러의 연마제가 잘 제거되었는지 확인하는 방법 알려줘

    Negative examples (These should NOT be classified as 1)
    Q1: 세제 추천해줘
    Q2: 세척도구 추천해줘
    Explanation: This is not clarification as the user is simply asking two very different objects. Therefore this should not be classified as clarification but rather narrowing down..

    Q1: 대전에 칠순잔치 할 만한 식당 추천해줘
    Q2: 대전 중구에서 칠순잔치 할 만한 식당 추천해줘
    Explanation: These are strictly two different prompts, one asking 칠순잔치 in 대전 vs one asking in 대전 중구 so it is not clarification but rather narrowing down.

    2) 주제 탐색 (Domain exploration)
    Def: 사용자가 동일한 과제를 수행하기 위해, 주제에 대한 광범위한 이해를 추구하거나 질문하는 경우. This can also be ambiguous, when the topic from Q1 and Q2 are related but are not very close as to narrowing down or repeating the same question, this should be classified.
    EG: 
    Q1: 스테인레스 텀블러 연마제 제거방법
    Q2:  세제 추천해줘

    Q1: 허리에 좋은 운동 뭐가 있을까
    Q2: 요가나 필라테스중에 더 도움이 되는 쪽이 뭘까

    Q1: 허리에 좋은 운동 뭐가 있을까
    Q2: 앞서 설명한 방법 중 집에서 할 수 있는 방법과 과정 설명해줘

    Q1: 서울 금호동과 그 주변의 유명한 정형외과는 어떤 곳이 있나요?
    Q2: 허리디스크, 척추협착증환자 모임

    Q1: 스테인리스 텀블러 연마제가 제거되었는지 확인하는 방법
    Q2: 스테인리스 텀블러 연마제가 제거되지 않았을 때의 현상

    Q1: 세척도구 추천해줘
    Q2: 텀블러 연마제 제거 방법추천

    Q1: 스테인리스 텀블러 연마제를 제거하기 가장 쉬운 방법
    Q2: 스테인리스 텀블러 연마제가 제거되었는지 확인하는 방법
    Explanation: This is not 4. 구체화 but 2. 도메인 탐색 because the question it is asking is different from how to remove 연마제 to how to check if it has been removed. Therefore it is exploring the domain.

    Q1: 한국과학기술원에서 제일 가까운 스타벅스 위치 알려줘
    Q2: 어은동에서 제일 가까운 스타벅스 위치 알려줘
    Explanation: This is also included in domain exploration because the user is trying to explore other starbucks locations

    3) 절대적인 정보량의 부족 (Low Information)
    Def: 사용자가 동일한 과제를 수행하기 위해, 절대적으로 정보량이 부족해 이를 보충하고자 하는 경우. When Q2 is related to responses hasn't been fully answered, put this taxonomy as a possibility.
    EG:
    Q1: 요가를 해본적 있어?
    R: "요가를 해본 적이 있나요? - 예, 요가를 해본 적이 있습니다. - 요가는 몸과 마음의 조화를 이루는 운동으로, 꾸준한 수련을 통해 몸과 마음의 균형을 맞추는 것이 중요합니다. - 요가는 인도에서 시작된 운동으로, 현재는 전 세계적으로 인기 있는 운동 중 하나입니다. - 요가를 하기 위해서는 특정한 자세와 호흡법을 따라야 하며, 이를 통해 몸과 마음의 건강을 추구합니다
    Q2: 요가를 하려면 어디로 가야할까

    Q1: 스테인리스 텀블러 연마제 어떻게 제거하지?
    R: 스테인리스 텀블러를 사용하기 전에 연마제를 제거하는 것이 좋습니다. 연마제는 스테인레스와 같은 금속 또는 유리의 표면을 깎아 부드럽게 만들기 위해 사용하는 재료입니다 15. 연마제의 주성분인 탄화규소를 통해 표면이 매끈해진 금속은 반짝반짝 광택을 얻게 됩니다.
    Q2: 첫번째 방법을 사용하면 안되는 제품은 어떤 것이야?
    Explanation: This is because in this response, there wasnt enough information saying what products we can use

    Q1: 위스키는 어때
    R: 오레포스 Orrefors 온더락 위스키 신혼부부 집들이 선물 글라스 시티 올드 x4 글래스 시티 올드 4개 세트로 구성된 제품입니다. 온더락은 위스키를 따뜻하게 마시는 방법으로, 신혼부부의 집에서도 쉽게 즐길 수 있습니다
    Q2: 온더락이 뭐야
    Explanation: This is considered to be 3 because in the response, it mentions 온더락 but doesnt full explain it. Therefore the user is asking for additional information 


    4) 구체화 (Narrowing Down)
    Def: 사용자가 동일한 의도를 가지고, Follow-up query를 통해 자세하고 구체적인 답변을 얻고자 하는 경우. In this method Q2 needs to specifically narrow down or scope down from the same question from Q1. If it is a similar topic but something different, it should be classified as 2.
    EG:
    Q1: 여름에 부산 가면 놀거리
    R: 부산의 여름에는 다음과 같은 놀거리가 있습니다...
    Q2: 해수욕장을 제외한 여름 부산 놀거리

    Q1: 두번째 차례에 일어난 쿠데타에 대해 더 자세히 알려줘
    R: 두 번째 차례의 쿠데타란, 2022년 10월 30일 서아프리카 부르키나파소에서 발생한 군사 쿠데타를 말합니다...
    Q2: 2023년 6월 24일에 일어난 러시아 쿠데타에 대해 더 자세히 알려줘

    Q1: 커피머신 중에서 가장 고급스러운 커피머신은 뭐야?
    Q2: 더 비싸고 좋은 커피머신을 알려줘. 그리고 캡슐 커피 머신 종류는 제외해서 알려줘

    Q1: 여름에 부산에 여행가려고 하는데, 어떤 일정이 좋을까?
    Q2: 여름에 부산 가면 놀거리

    Q1: 특별한 신혼부부 집들이 선물 종류는 뭐가 있지
    Q2: 해외 전자제품 종류 중에서 추천해줘
    Explanation: The user is scoping down to 해외 전자제품 Therefore it is narrowing down.

    Q1: 신혼부부 집들이 선물로 휴지 들고가도 돼?
    Q2: 너무 비싸 5천원 이하 집들이 선물 알려줘
    Explanation: The user wants to narrow down to under a certain price therefore it is narrowing down.

    Negative Examples 

    Q1: 삼성식기세척기 50만원대 알려줘
    Q2: 비스포크 dw50 장점
    Explanation: Although Q1 and Q2 are pretty related, they are asking a different scope and therefore is not narrowing down on the same topic. 

    Q1: 신혼부부 집들이 선물에 적합한 가전제품 추천해줘
    Q2: 커피머신 중에서 가장 고급스러운 커피머신은 뭐야?
    Explanation: Although Q1 and Q2 are pretty related, they are asking a different scope and therefore is not narrowing down on the same topic. 


    5) 답변 형식 변환 (Answer Format Conversion)
    Def: 검색엔진이 제공한 내용을 특정 형식(예: 목록, 표, 시각적 자료 등)으로 제공받기를 원하는 경우. It has to be on a different format.
    EG: 
    Q1: 대전 유성구 어은로 스타벅스에 가는 방법을 물어볼게
    R: 대전 유성구 어은로 스타벅스에 가는 방법은 다음과 같습니다...
    Q2: 유성구 어은 스타벅스를 지도에서 위치를 표시해줄 수 있니?

    Q1: 방콕에서 여행갈만한 장소 알려줘
    Q2: 구글에서 검색된 자료들도 같이 보여줘

    Q1: 숙취해소제 아이솔솔에 대해 알려줘
    Q2: 아이솔솔의 최저가와 링크를 알려줘


    6) 정보 검증 (Verification)
    Def: 검색엔진이 제공한 내용에 대해 추가적인 증거나 확인을 요청하는 경우. This is also very ambiguous, if you feel like the Q2 is verifying a certain information from R, include it in the possible sets.
    EG: 
    Q1: 숙취에 두통약 먹어도 돼?
    R: 숙취에 두통약 먹어도 되나요? - 숙취로 인한 두통은 아세트알데히드와 같은 독성 물질 때문에 생길 수 있습니다...
    Q2: 숙취로 인한 두통에 먹을 수 있는 약은 없어?

    Q1: 부부합산 연 소득이 5천만원 초과하는 경우 정부지원 전세대출 방법 알려줘
    Q2: 부부합산 소득이 연 5천만원을 초과하는 경우 받을 수 있는 정부지원 전세대출은 없는거야?

    Q1: 50만원대 식기세척기를 사고싶어, R: 50만원대 식기세척기 제품들을 추천해드리겠습니다
    Q2: 너가 알려준 3가지 모델 중에 50만원대가 없는데?

    Q1: 스테인리스 말고 쇠로 된 텀블러 연마제는 어떻게 제거하지?,
    Q2: 1번부터 6번까지 순서대로 진행해야 되는 거야?

    Negative examples

    Q1: 숙취에 두통약 먹어도 돼?
    R: 숙취에 두통약을 먹어도 되는지 궁금하신 것 같습니다. 숙취로 인한 두통은 아세트알데히드라는 독성 물질이 뇌혈관을 확장시키면서 생기는 것으로, 아세트알데히드를 분해해야 합니다 12. 따라서 숙취 두통약으로는 아세트아미노펜 성분이 아닌, 아스피린이나 비스테로이드 항염증제(예: 이부프로펜, 나프록센 등) 
    Q2: 숙취에 타이레놀은 도움이 될까?
    Explanation: In no where does the reply mention 타이레놀 therefore this is not verifying the previous response, rather this should be domain exploration. The Q2 has to specifically ask question regarding the response for it to be verification.

    7) 리액션 
    Def: 검색 엔진이 제공한 답변에 대한 만족이나 불만족의 감정을 표현한 경우
    EG: 
    Q1: 주택청약 통장에 든 돈이 내가 당첨된 아파트 가격보다 적으면 어떡해
    R: 주택청약 통장에 든 돈이 내가 당첨된 아파트 가격보다 적으면 어떡하나요? - 주택청약 통장에 가입하고 나서 일정 기간 동안 일정 금액을 납입하면,...
    Q2: 아하 그렇군!

    Q1: 너가 알려준 3가지 모델 중에 50만원대가 없는데?
    Q2: 갤럭시탭 말고 식기세척기 말이야,,, 정신차리자 큐야

    Q1: 로봇청소기 말고 식기세척기!
    Q2: 정확히 모르겠으면 나한테 되물어서 질문 의도를 제대로 파악하는 게 어때?


    8) 분류 불가
    Def: 질문의 의도나 내용이 불분명한 경우. Q2 must not be related to both Q1 and R.
    EG: 
    Q1: 어은동에서 제일 가까운 스타벅스 위치 알려줘
    Q2: 외국 여행 가면 어떤 종류의 치즈를 사올 수 있어?
    Explanation: This should only be choosen as 8 if it is asking very unrelated. If Q1 and Q2 are implicitly related it is Not 8

    Negative examples:
    Q1: 5만원 대의 신혼부부 집들이 선물 골라줘
    Q2: 위스키는 어때 
    Explanation: There might be examples of when Whisky is asked for the present therefore this is NOT 8.


   Based on these definitions and examples, categorize the following conversation into the specific subcategory that is all labeled with a number:
   Also keep in mind of the initial query before you categorize something as 8

   Initial Task: {initial_intent}
   Q1: {previous_turn}
   R: {response}
   Q2: {current_turn}

   Only give the output as the possible number or numbers that correlates with the subcategories(1~8) You can give at most three numbers if it fits more than one. separate the possible numbers with comma (EX: 1,2,3). Dont break the format as it is crucial because we are using api to get the data automatically.
  
   """

    try:
        response = client.chat.completions.create(
            model="",
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
    if (
        row["turn_no"]
        > dataset.loc[
            dataset["conversation_id"] == row["conversation_id"], "turn_no"
        ].min()
    ):
        try:
            # Getting the initial intent for the conversation
            initial_intent = dataset.loc[
                (dataset["conversation_id"] == row["conversation_id"])
                & (
                    dataset["turn_no"]
                    == dataset.loc[
                        dataset["conversation_id"] == row["conversation_id"], "turn_no"
                    ].min()
                ),
                "contextual_question",
            ].iloc[0]

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
            print(f"Initial Intent: {initial_intent}")
            print(f"Q1: {previous_turn}")
            # print(f"R: {response}")
            print(f"Q2: {current_turn}")

            taxonomy_number = analyze_conversation(
                initial_intent, previous_turn, current_turn, response
            )
            print(f"Taxonomy Number: {taxonomy_number}")

            # Update the current row's taxonomy based on the analysis
            dataset.at[index, "taxonomy"] = taxonomy_number
        except IndexError:
            # Handle cases where the previous turn is not found
            print(f"Previous turn not found for index {index}")

# Saving the modified dataset
dataset.to_csv("Full Results/Axis1_filtered_conversations2_50.csv", index=False)
