# from pykospacing import Spacing
from collections import Counter
from konlpy.tag import Okt
from hanspell import spell_checker
from youtube_transcript_api import YouTubeTranscriptApi
from oauth2client.tools import argparser
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import nltk
import pafy
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, notebook

from torch.nn import init
import gc
import unicodedata
import re

import requests
import pprint
import json
import sys

print('im started')
class BERTDataset(Dataset): #데이터셋을 BERT 임베딩하는 API. 데이터가 KoBERT 모델의 입력으로 들어갈 수 있는 형태가 되도록 토큰화, 정수 인코딩, 패딩 등을 해주어야 한다. SKT에서 제공한 API이기때문에 별도의 수정은 필요없습니다.
    def __init__(self, data, bert_tokenizer, max_len, pad, pair): # parameter: text data, Tokenizer, 최대 sequence 길이, padding, pair
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair) #BERTSentenceTransform 이라는 모듈을 사용하여

        self.sentences = []

        if len(data) <= max_len:
            self.sentences.append(transform([data]))
        else:
            self.sentences.append(transform([data[:max_len]]))

    def __getitem__(self, i):
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))

'''
Parameters : sent (str) – input sentence to be sentence embedded

Returns : embedded sentence array

Return type : <class '__main__.BERTDataset'> [np.arrays]
'''

class BERTClassifier(nn.Module): #SKT에서 제공한 API이므로 num_classes 이외에 수정할 필요가 없습니다
    def __init__(self, bert, hidden_size=768, num_classes=11, dr_rate=None, params=None): #받아온 pretrained 모델과 히든사이즈, nn.linear에 클래스 개수만큼의 출력값을 내게끔 num_class를 설정한다. num_class는 분류할 클래스 개수입니다.
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, 
                              token_type_ids=segment_ids.long(), 
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

'''
Parameters : sent (str) – sentence to be classified
             valid_length - attention mask
             segment_ids – candidate labels

Returns : confidence scores corresponding to each input label

Return type : <class 'torch.Tensor'>
'''

def GetMediaCategory(text): #youtube 에서 추출한 string 데이터가 parameter로 들어갑니다. Kobert모델에 입력되어 num class만큼의 tensor가 출력됩니다. 출력된 tensor중 가장 높은 tensor의 index와 weight값을 반환한다.
    text = unicodedata.normalize('NFC', text)
    text = ' '.join(re.compile('[가-힣]+').findall(text))
    if len(text) == 0:
        text = '기타'

    data = BERTDataset(text, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data, 
                                                  batch_size=batch_size, 
                                                  num_workers=num_workers)
    gc.collect()
    wholeout = []
    wholevalue = []

    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        outlist = []
        valuelist = []
        # 모든 카테고리에 대해서 Value값 도출
        out = modelbest(token_ids, valid_length, segment_ids)
        for outi in out:
            valuelist.append(outi.max().tolist())
            # 설정한 Threshold 값보다 크면 해당 카테고리로 분류
            if outi.max().tolist() > threshold:
                outlist.append(categorylist[outi.argmax()])
            # 아니면 기타 카테고리로 분류
            else:
                outlist.append('기타')
        wholeout += outlist
        wholevalue += valuelist

    return wholeout, wholevalue

'''
Parameters : sent(str) - input query to be classified from youtube

Returns : label corresponding to max confidence score and score itself

Return type : list, list
'''
###################################################################################
###################################################################################
###################################################################################


## GPU 사용 시
# device = torch.device("cuda : 0")

## CPU 사용 시
device = torch.device('cpu')
gc.collect()

# BERT 모델 및 Tokenizer 선언
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 형태소 분석을 위한 Konlpy 변수 생성
okt = Okt()

# Define Parameters
max_len = 512   # Bert 모델 Max Sequence Length Size
batch_size = 6
warmup_ratio = 0.1  # Warmup 비율
num_epochs = 20
max_grad_norm = 1
log_interval = 20   # Log 간격 수
learning_rate = 5e-6   # 학습률 (e.g., 5e-5, 2e-5)
# num_workers = 2
num_workers = 0
n_splits = 5    # 교차 검증 분할 사이즈

# 분류할 대표 카테고리 (11가지)
categorylist = ["화장품", "패션", "요리음식", "여행아웃도어", "인테리어",
                "엔터테인먼트", "육아", "아이티", "자동차", "헬스/피트니스", "반려동물"]
# 설정한 Threshold (대표 카테고리 선정 기준값)
threshold = 5.26

# Pre-trained Model
# Local에서 구동 시 모델이 있는 경로 재설정 필요
modelbest = torch.load(
            "C:/Users/ASUS/Projects/lab/Eden Alice Marketing/Workspace/kobertbest_512.pt",
            map_location=device)
modelbest.to(device)
modelbest.eval()

###################################################################################


# URL에 대한 제목과 태그 데이터 추출
def GetYoutubeData(youtube_url): #유튜브 url을 입력파라미터로 넣어 youtube API를 통해 메타데이터를 반환한다.
  url = youtube_url
  DEVELOPER_KEY = "AIzaSyC1yBL6YbPZj5nwrtDa0tlXa6-7A3Ur5B8"

  video = pafy.new(url)
  ID = video.videoid
  thumbnail = video.thumb
  title = video.title   # 제목 데이터 추출

  response = requests.get("https://www.googleapis.com/youtube/v3/videos?&part=snippet&key=" 
                          + DEVELOPER_KEY + "&id=" + ID)
  snippet = response.json()["items"][0]['snippet']

  # 태그 데이터 추출
  tag_data = None
  if 'tags' in snippet:
    tag = snippet['tags']
    tag_data = ' '.join(tag)
    title += tag_data
  else:
    return title

  return title

'''
Parameters : sent(str) - youtube url to extract title and hashtag

Returns : title and hashtags from youtube

Return type : string
'''
###################################################################################

# URL에 대한 자막 데이터 처리
def GetYoutubeCaption(youtube_url, slicing_parameter): #유튜브 url을 넣어 subtitle의 토큰을 반환한다.
  url = youtube_url
  video = pafy.new(url)
  ID = video.videoid
  caption = ''
  flag = 1

  # 자막 데이터 받아오기
  try:
    srt = YouTubeTranscriptApi.get_transcript(ID, languages=['ko'])  # 한국말 자막
  except:
    try:
      srt = YouTubeTranscriptApi.get_transcript(
          ID, languages=['en'])  # 없으면 영어 자막
      flag = 2
    except:
      flag = 3
      srt = []
      print("Doesn't have a transcript")

  for i in srt:
    caption += i['text'] + ' '

  word_list = caption.split(' ')
  result = {}

  # 형태소 분석
  morph = []

  for word in word_list:
    morph.append(okt.pos(word))

  # 명사(토큰화) 리스트
  noun_list = []
  # 불용어 리스트 (추가하기)
  stopword_list = ['제', '제가', '뭐', '진짜', '안녕', '은', '는', '이', '가', '을', '를', '와', 
                  '과', '도', '에', '에서', '의', '거', '그', '또', '것', '그리고', '더', '아', 
                  '좀', '뭐', '정말', '많이', '한', '이렇게', '수', '우리', '이제', '때', '저',
                  '제품', '추천', '사용', '느낌', '분', '생각', '지금', '쪽', '살짝', '약간',
                  '여기', '안', '처리', '일단', '경우', '정', '의미', '다음', '되게', '요', '게',
                  '점', '때문', '정도', '한번', '요런', '이런', '그냥', '해', '예', '가지', '사실',
                  '중', '기능', '오늘', '구매', '설명', '오', '음', '대해', '왜', '하나', '가요',
                  '자', '리뷰', '바로', '시작', '얼마나', '사람', '박수', '역시', '무슨', '아따',
                  '놈', '오', '오오', '오오오', '정리', '말', '으르렁', '아', '아아', '아아아',
                  '너', '나', '총', '방법', '어제', '내', '네', '못', '어디', '뒤', '구', '막',
                   '음악', '무조건', '이번', '항상', '완전', '조금', '얘', '여러분', '치', '카',
                   '후', '끼', '걸', '저기', '보고', '처', '저희', '번', '거기', '웬', '이건',
                   '저희', '제일', '마무리', '대신', '잡', '잔뜩', '실컷', '멀리', '녀석', '곳',
                   '온', '그것', '후회', '럼', '두', '후기', '댓글', '참고', '상세', '평점', '인',
                   '로', '소개', '사이트', '수가', '드릴', '보', '단', '추가', '마음', '버전',
                   '흡입', '친구', '조절', '과정', '법', '물', '거품', '다시', '밤', '양', '전',
                   '노', '장점', '뉴', '데', '뚜껑', '미리', '고객', '가격', '차이', '일반', '단점',
                   '부분', '별로', '원래', '구입', '제대로', '웃음', '앞', '광고주', '상태', '아주',
                   '어', '먼저', '구독', '난', '악', '다른', '날', '만', '없', '우린', '저희']

  # 자막 데이터 중 불용어 리스트에 포함되지 않는 요소 선별
  for sentence in morph:
      for word, tag in sentence:
          if tag in ['Noun'] and word not in stopword_list:
            noun_list.append(word)

  count = Counter(noun_list)
  words = dict(count.most_common())
  wordclouds = words

  # Top-N 토큰화
  keyword_list = []
  for i in words:
      keyword_list.append(i)
  keyword_extraction = (keyword_list[0:slicing_parameter])

  return keyword_extraction, wordclouds

'''
Parameters : sent(str) - youtube url to extract keywords from subtitle

Returns : keywords from youtube subtitle

Return type : list
'''
# 각 상황에 따른 카테고리 분류 결과 출력
def CheckData(text, keyword_extraction): #추출한 메타데이터를 학습된 Kobert모델에 입력으로 넣어 클래스를 반환받는다.
  answer_class = None
  answer_value = 0
  spacing = Spacing()

  title = GetYoutubeData(text)
  print('제목과 해시태그 추출 :', title)
  # 오리지널 제목, 해시태그 들어갔을 때 결과 도출
  title_classlist, title_valuelist = GetMediaCategory(title)
  print('모델에 넣은 결과 :', title_classlist[0], title_valuelist)
  print('\n')
  answer_class = title_classlist
  answer_value = title_valuelist

  # Subtitle 토큰값이 들어갔을 때 결과 도출
  keyword = None
  keyword = ' '.join(keyword_extraction)
  print('subtitle에서 추출한 top-n 토큰들 :', keyword)
  tag_classlist, tag_valuelist = GetMediaCategory(keyword)
  print('모델에 넣은 결과 :', tag_classlist[0], tag_valuelist)
  if tag_valuelist > answer_value:
    answer_value = tag_valuelist
    answer_class = tag_classlist

  multi = title + keyword
  spaced_multi = spacing(multi)

  # 오리지널 제목, 헤시태그, Subtitle 토큰값이 들어갔을 때 결과 도출
  print('\n혼합된 결과 :', spaced_multi)
  multi_class, multi_valuelist = GetMediaCategory(spaced_multi)
  print('모델에 넣은 결과 :', multi_class[0], multi_valuelist)
  if multi_valuelist > answer_value:
    answer_value = multi_valuelist
    answer_class = multi_class

  return answer_class, answer_value

'''
Parameters : sent (str) - youtube url
             list (str) - extracted keywords  

Returns : label corresponding to max confidence score and score itself 

Return type : list, list
'''
###################################################################################
###################################################################################
###################################################################################


# 코드 실행 명령어에서 YouTube URL값 받아옴
target_url = sys.argv[1]

# 입력받은 YouTube URL에 대한 자막 데이터에서 Top-N 토큰화 실행
# Top-N에서의 N값 파라미터로 설정 가능
keyword_extraction = GetYoutubeCaption(target_url, 5)
print(keyword_extraction[0], end="\n\n")

# 입력받은 YouTube URL에 대한 카테고리 분류
classlist, valuelist = CheckData(target_url, keyword_extraction[0])

# 결과값 확인
print(classlist)
print(valuelist, end="\n\n")


# 최종 결과값 텍스트 파일로 저장
with open("result.txt", "w") as f:
  f.write(classlist[0])
  f.write(' ' + str(valuelist[0]))
