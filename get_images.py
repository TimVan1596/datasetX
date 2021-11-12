import os
import urllib
import urllib.request
from bs4 import BeautifulSoup
import random
import time
from urllib.parse import urlparse

# 收集到的常用Header
my_headers = [
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "
]

cookies = {'cookies': ''}


# 根据url获取文件名和后缀
# 返回 （文件名,后缀）
def get_url_file_info(url):
    file_name = os.path.basename(urlparse(url).path)
    return os.path.splitext(file_name)


# 获取图片
# keyword = 关键词
# path = 图片待保存的路径
def getImage(keyword, path):
    # 这里拼凑成一个url
    url = 'https://699pic.com/tupian/' + keyword + '.html'
    # 这里获取网站的源码，源码主要是很长的一段html格式的字符串

    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    req = urllib.request.Request(url, headers={
        'User-Agent': random.choice(my_headers)
    })

    response = urllib.request.urlopen(req)
    content = response.read().decode('utf-8')

    # BeautifulSoup这个库可以解析html格式的字符串，把网页的源码解析成一个个类，然后你就可以依次去访问它#的属性，比如head，div，src等等
    soup = BeautifulSoup(content, 'html.parser')
    # 这里就是对BeautifulSoup的运用了
    all_a = soup.find_all('div', class_='list')
    print("关键词为{1}，共检测到{0}张图片".format(len(all_a), keyword))
    for i in range(len(all_a)):
        img_url = 'https:' + str(all_a[i].img.attrs['data-original'])
        file_name, file_ext = get_url_file_info(img_url)
        # 替换文件名中的点
        file_name = file_name.replace(".", "_")
        # 防止重名
        millis = str(round(time.time() * 1000))
        filename = "{0}_{1}_{2}{3}".format(i, millis, file_name, file_ext)
        # 拼接文件名
        file_full_name = '{0}{1}'.format(path, filename)
        urllib.request.urlretrieve(img_url, filename=file_full_name)
        print("{0}=>{1}".format(i, file_full_name))
    print("保存完成")


if __name__ == '__main__':
    keyword = 'renwu'
    path = './origin/test/'
    getImage(keyword, path)
