#!/usr/bin/env python3
"""
Autodl私有云Authorization Token获取脚本
专门用于获取JWT authorization token

使用方法:
python get_auth_token.py

返回完整的Authorization token，可直接用于API调用
"""

from collections import Counter
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
import json
import base64
import time
import random
from datetime import datetime


class AutodlTokenGetter:
    def __init__(self, phone, password, login_url, chrome_driver_path):
        self.phone = phone
        self.password = password
        self.option = Options()
        self.option.add_argument("--headless=new")  # 无头模式
        self.option.add_argument("--enable-logging")
        self.option.add_argument("--log-level=0")
        self.option.add_argument("--no-sandbox")
        self.option.add_argument("--disable-dev-shm-usage")
        self.option.add_experimental_option('useAutomationExtension', False)
        self.option.add_experimental_option("excludeSwitches", ["enable-automation"])
        # 启用性能日志记录（移除了无效的enableTimeline选项）
        self.option.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': False
        })
        self.chrome_driver_path = chrome_driver_path
        # 设置日志级别以捕获网络请求
        self.option.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
        self.login_url = login_url

    def get_token(self, max_retries=3):
        """获取Authorization Token（带重试机制）"""
        for attempt in range(max_retries):
            if attempt > 0:
                wait_time = random.randint(10, 20)
                print(f"⏳ 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

            print(f"🔄 尝试 {attempt + 1}/{max_retries}")
            token = self._get_token_single_attempt()
            if token:
                return token

        return None

    def _get_token_single_attempt(self):
        """单次获取token的尝试"""
        self.driver = webdriver.Chrome(
            service=Service(self.chrome_driver_path),
            options=self.option
        )
        self.driver.get(self.login_url)
        sleep(2)

        # 查找手机号输入框并输入手机号
        username_input = self.driver.find_element(By.XPATH, "//input[@placeholder='请输入手机号']")
        username_input.send_keys(self.phone)

        # 查找密码输入框并输入密码
        password_input = self.driver.find_element(By.XPATH, "//input[@placeholder='请输入密码']")
        password_input.send_keys(self.password)

        # 清除之前的日志
        self.driver.get_log('performance')

        # 查找登录按钮并点击（使用多种定位策略）
        login_button = self.driver.find_element(By.XPATH, "//button[.//span[normalize-space()='登录']]")
        login_button.click()

        sleep(2)

        logs = self.driver.get_log('performance')
        self.driver.quit() # 退出浏览器
        auth_headers = []
        for log in logs:
            message = json.loads(log['message'])
            request_headers = message['message']['params'].get('request', {}).get('headers', {})
            token = request_headers.get('Authorization') or request_headers.get('authorization')
            if token and token != "null":
                auth_header = request_headers.get('Authorization') or request_headers.get('authorization')
                if auth_header and auth_header != "null":
                    auth_headers.append(auth_header)

        if auth_headers:
            auth_counter = Counter(auth_headers)
            most_common_auth = auth_counter.most_common(1)[0][0]
            print("✅ 成功获取token!")
            self._print_token_info(most_common_auth)
            return most_common_auth

        return None

    def _print_token_info(self, jwt_token):
        """打印token信息"""
        try:
            payload_part = jwt_token.split('.')[1]
            payload_part += '=' * (4 - len(payload_part) % 4)
            payload = json.loads(base64.b64decode(payload_part))

            if 'exp' in payload:
                expiry_timestamp = payload['exp']
                expiry_time = datetime.fromtimestamp(expiry_timestamp)
                print(f"📅 过期时间: {expiry_time}")

                # 计算剩余时间
                remaining = expiry_time - datetime.now()
                hours = remaining.total_seconds() / 3600
                print(f"⏰ 剩余有效时间: {hours:.1f} 小时")
        except:
            print("⚠️ 无法解析token信息")


def main():
    """主函数 - 直接运行获取token"""
    print("🔐 Autodl私有云Authorization Token获取工具")
    print("=" * 60)

    # 您的账号信息
    phone = "账号"
    encrypted_password = "密码"
    login_url = "https://www.autodl.com/login?private_cloud_url=https%3A%2F%2Fprivate.autodl.com%2Flanding%3Furl%3D%2Flanding"
    chrome_driver_path = "/home/jh/chromedriver-linux64/chromedriver"

    # 创建获取器
    token_getter = AutodlTokenGetter(phone, encrypted_password, login_url, chrome_driver_path)

    # 获取token
    print("🚀 开始获取Authorization Token...")
    token = token_getter.get_token()

    if token:
        print("\n" + "=" * 60)
        print("🎉 获取成功！您的Authorization Token:")
        print("=" * 60)
        print(token)
        print("=" * 60)

        # 验证token
        print("\n🔍 验证token有效性...")
        session = requests.Session()
        test_url = "https://private.autodl.com/admin/v2/user/list"
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
        }

        response = session.post(test_url, json={"page_index": 1, "page_size": 1}, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 'Success':
                print("✅ Token验证成功！可以直接用于API调用")
                print("\n💡 使用示例:")
                print('headers = {"Authorization": "' + token + '"}')
                print('response = requests.post(api_url, json=payload, headers=headers)')
            else:
                print(f"⚠️ Token验证失败: {result.get('msg')}")
        else:
            print(f"⚠️ 验证请求失败: {response.status_code}")
    else:
        print("\n❌ 获取Authorization Token失败")
        print("💡 建议: 请稍后重试，或检查网络连接")


if __name__ == "__main__":
    main()
