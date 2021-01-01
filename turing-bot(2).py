import json
import urllib.request


def get_answer(text_input):
    api_url = "http://openapi.tuling123.com/openapi/api/v2"

    req = {
        "perception":
            {
                "inputText":
                    {
                        "text": text_input
                    },

                "selfInfo":
                    {
                        "location":
                            {
                                "city": "北京",
                                "province": "北京",
                                "street": "信息路"
                            }
                    }
            },

        "userInfo":
            {
                "apiKey": "f50c9da915a0434988227175721e459f",
                "userId": "OnlyUseAlphabet"
            }
    }
    # print(req)
    # 将字典格式的req编码为utf8
    req = json.dumps(req).encode('utf8')
    # print(req)

    http_post = urllib.request.Request(api_url, data=req, headers={'content-type': 'application/json'})
    response = urllib.request.urlopen(http_post)
    response_str = response.read().decode('utf8')
    # print(response_str)
    response_dic = json.loads(response_str)
    # print(response_dic)

    intent_code = response_dic['intent']['code']
    results_text = response_dic['results'][0]['values']['text']
    # print('code：' + str(intent_code))
    # print('text：' + results_text)
    return results_text


def chat_robot():
    """
    聊天机器人
    :return:
    """
    # 这个变量存储的是机器人给用户的响应信息;
    response = ''
    while True:
        # response :机器人给用户的响应信息
        # receive： 接收用户 给 机器人传来的消息
        receive = yield response
        response = get_answer(receive)


if __name__ == '__main__':
    # 实例化 Robot是生成器
    my_robot = chat_robot()
    # 调用next方法，遇到yield 停止
    next(my_robot)

    while True:
        send_date = input("[User]>>:")
        if send_date == 'q' or send_date == 'quit':
            print("goodbye")
            break
        # 将用户输入的send_date传送到yield 前的receive
        response = my_robot.send(send_date)
        print('[Robot>>:]', response)
