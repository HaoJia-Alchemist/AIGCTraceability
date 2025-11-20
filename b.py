import json
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from collections import Counter

# 配置Chrome选项以启用网络日志记录
options = Options()
options.add_argument("--enable-logging")
options.add_argument("--log-level=0")
options.add_experimental_option('useAutomationExtension', False)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
# 启用性能日志记录（移除了无效的enableTimeline选项）
options.add_experimental_option('perfLoggingPrefs', {
    'enableNetwork': True,
    'enablePage': False
})

# 设置日志级别以捕获网络请求
options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# 访问指定网址
url = "https://www.autodl.com/login?private_cloud_url=https%3A%2F%2Fprivate.autodl.com%2Flanding%3Furl%3D%2Flanding"
driver.get(url)

# 等待页面加载
sleep(2)

# 查找手机号输入框并输入手机号
username_input = driver.find_element(By.XPATH, "//input[@placeholder='请输入手机号']")
username_input.send_keys("18560307751")

# 查找密码输入框并输入密码
password_input = driver.find_element(By.XPATH, "//input[@placeholder='请输入密码']")
password_input.send_keys("hj19990930")

# 清除之前的日志
driver.get_log('performance')

# 查找登录按钮并点击（使用多种定位策略）
# 方法1: 通过span文本定位
login_button = driver.find_element(By.XPATH, "//button[.//span[normalize-space()='登录']]")
login_button.click()

print("手机号和密码已输入，已点击登录按钮")

# 等待登录结果和网络请求
sleep(5)

# 获取网络请求日志
print("\\n网络请求列表:")
logs = driver.get_log('performance')
request_count = 0
auth_headers = []

for log in logs:
    message = json.loads(log['message'])
    method = message.get('message', {}).get('method', '')

    request_count += 1
    print(f"{request_count}. {method}")
    print(json.dumps(message['message'], indent=2, ensure_ascii=False))
    print("-" * 50)

    request_headers = message['message']['params'].get('request', {}).get('headers', {})
    auth_header = request_headers.get('Authorization') or request_headers.get('authorization')
    if auth_header and auth_header != "null":
        auth_headers.append(auth_header)
        print(f"Authorization header: {auth_header}")
        print("=" * 50)

print(f"\\n总共捕获到 {request_count} 个网络请求/响应")

# 统计并找出出现次数最多的Authorization header
if auth_headers:
    auth_counter = Counter(auth_headers)
    most_common_auth = auth_counter.most_common(1)[0]
    print(f"\\n收集到 {len(auth_headers)} 个有效的Authorization头部")
    print(f"出现次数最多的Authorization header: {most_common_auth[0]} (出现次数: {most_common_auth[1]})")
else:
    print("\\n未收集到有效的Authorization头部")

# 如果需要，可以在这里添加更多操作...
driver.quit()