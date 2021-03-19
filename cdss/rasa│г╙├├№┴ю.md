### 1. 基本命令

#### 1.1 训练模型

无论是单独训练，还是2个模型一起训练，均可以使用rasa train命令实现（rasa nlu和rasa core合并为RASA），不同之处在于，提供的训练数据，若同时提供两份数据，则会同时训练nlu和core。

* 训练nlu模型

  > ```shell
  > python -m rasa train --config configs/zh_jieba_supervised_embeddings_config.yml --domain configs/domain.yml --data data/nlu/number_nlu.md data/stories/number_story.md
  > ```

* 训练core模型

  > ```shell
  > python -m rasa train --config configs/zh_jieba_supervised_embeddings_config.yml --domain configs/domain.yml --data data/stories/number_story.md
  > ```

* 一起训练两个模型

  > ```shell
  > python -m rasa train --config configs/zh_jieba_supervised_embeddings_config.yml --domain configs/domain.yml --data data/nlu/number_nlu.md data/stories/number_story.md
  > ```

  

#### 1.2 启动rasa

* 不额外独立nlu服务

  > ```shell
  > python -m rasa run --port 5005 --endpoints configs/endpoints.yml --credentials configs/credentials.yml 
  > ```

* 启动额外nlu服务

  > ```shell
  > python -m rasa run --port 5005 --endpoints configs/endpoints.yml --credentials configs/credentials.yml --enable-api -m models/nlu-20190515-144445.tar.gz 
  > ```

* 单独启动nlu服务

  > ```shell
  > rasa run --enable-api -m models/nlu-20190515-144445.tar.gz
  > ```

* 调用第三方模型服务

  > ```shell
  > rasa run --enable-api --log-file out.log --endpoints my_endpoints.yml
  > ```
  >
  > 在my_endpoints.yml中配置rasa模型的url
  >
  > ```json
  > models:
  >   url: http://my-server.com/models/2019-1222.tar.gz
  >   wait_time_between_pulls: 10   # [optional](default: 100)
  > ```
  >
  > rasa模型还可以存放在[S3](https://aws.amazon.com/s3/) , [GCS](https://cloud.google.com/storage/) and [Azure Storage](https://azure.microsoft.com/services/storage/) 云盘中，同样远程调用，这里不多赘述。
  >
  > <https://rasa.com/docs/rasa/user-guide/configuring-http-api/>



nlu服务开启后，post方式调用

post http://localhost:5005/model/parse

```json
input:
{"text": "明天成都的天气如何"}
output:
{
    "intent": {
        "name": "request_weather",
        "confidence": 0.5547698674
    },
    "entities": [
        {
            "entity": "date_time",
            "value": "明天",
            "start": 0,
            "end": 2,
            "confidence": null,
            "extractor": "MitieEntityExtractor"
        },
        {
            "entity": "address",
            "value": "成都",
            "start": 2,
            "end": 4,
            "confidence": null,
            "extractor": "MitieEntityExtractor"
        }
    ],
    "intent_ranking": [
        {
            "name": "request_weather",
            "confidence": 0.5547698674
        },
        {
            "name": "affirm",
            "confidence": 0.0898880708
        },
        {
            "name": "deny",
            "confidence": 0.078563989
        },
        {
            "name": "thanks",
            "confidence": 0.072717722
        },
        {
            "name": "goodbye",
            "confidence": 0.0726876305
        },
        {
            "name": "greet",
            "confidence": 0.0697155938
        },
        {
            "name": "whattodo",
            "confidence": 0.0443638481
        },
        {
            "name": "whoareyou",
            "confidence": 0.0172932785
        }
    ],
    "text": "明天成都的天气如何"
}
```



#### 1.3 启动action

> ```shell
> python -m rasa run actions --port 5055 --actions actions --debug
> ```

### 2. NLU

![1590138247599](assets/1590138247599.png)

![1590138270872](assets/1590138270872.png)





#### 2.1 数据格式

rasa支持markdown和json两种数据格式，同时提供了相互转化的命令。

nlu训练数据包含四个部分：

* Common Example

  唯一必须提供数据。由三部分组成：intent、text、entities。

  > \#\# intent: 你的意图名称
  >
  > \- text

  text中可以不包括实体，如果有实体使用```[entityText](entityName)```来标记。

* synonyms

* Regular Expression Features

* lookup tables

#### 2.2 数据转化

rasa提供了json和markdown格式的转化命令

```
usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] --data DATA --out OUT
                             [-l LANGUAGE] -f {json,md}

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Path to the file or directory containing Rasa NLU
                        data. (default: None)
  --out OUT             File where to save training data in Rasa format.
                        (default: None)
  -l LANGUAGE, --language LANGUAGE
                        Language of data. (default: en)
  -f {json,md}, --format {json,md}
                        Output format the training data should be converted
                        into. (default: None)

Python Logging Options:
  -v, --verbose         Be verbose. Sets logging level to INFO. (default:
                        None)
  -vv, --debug          Print lots of debugging statements. Sets logging level
                        to DEBUG. (default: None)
  --quiet               Be quiet! Sets logging level to WARNING. (default:
                        None)
```

example：

> ```shell
> rasa data convert nlu --data M3-training_dataset_1564317234.json --out M3-training_dataset_1564317234.md -f md
> ```



### 3. core

备注1： core模型是一个分类分为，类别数为domain中定义的action数。action主要分为以下4类：

* custom action

* utterance：response、template

* default actions

  > ['action_listen', 'action_restart', 'action_session_start', 'action_default_fallback', 'action_deactivate_form', 'action_revert_fallback_events', 'action_default_ask_affirmation', 'action_default_ask_rephrase', 'action_back']

* forms（表单）

action类别为各类action取交集后的结果。



备注2：用户可以直接像core发送请求，不经过nlu模块。参数格式：/intent{"entityName": "entityValue", ...}

> ```json
> {
> 	"sender": "kkkkkkk",
> 	"message": "/search_treat{\"disease\":\"丛集性头痛\", \"sure\":\"丛集性头痛\"}"
> }
> ```
>
> ```
> [
>     {
>         "recipient_id": "kkkkkkk",
>         "text": "丛集性头痛的简介：从集性头痛(clusterheadaches)亦称偏头痛性神经痛、组胺性头痛、岩神经痛、蝶腭神经痛、Horton头痛等，.病员在某个时期内突然出现一系列的剧烈头痛。一般无前兆，疼痛多见于一侧眼眶或(及)额颞部，可伴同侧眼结膜充血、流泪、眼睑水肿或鼻塞、流涕，有时出现瞳孔缩小、垂睑、脸红、颊肿等症状。头痛多为非搏动性剧痛。病人坐立不安或前俯后仰地摇动，部分病员用拳击头部以缓解疼痛。"
>     },
>     {
>         "recipient_id": "kkkkkkk",
>         "text": "丛集性头痛的治疗方式有：药物治疗、支持性治疗"
>     }
> ]
> ```




