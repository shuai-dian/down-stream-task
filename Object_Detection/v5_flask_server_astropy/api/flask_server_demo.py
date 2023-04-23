from flask import Flask, request

app = Flask(__name__)


#将二进制流保存为文件
def save_file_from_byte(file_byte,filePath):
    with open(filePath, 'wb+') as f:
        f.write(file_byte)  # 二进制转为文本文件保存再本地


@app.route('/upload_fits', methods=['POST'])
def upload():
    if (request.method == "POST"):
        fileStorage = request.files['file'] # 二进制文件
        buffer_data = fileStorage.read()
        filename = request.files['textfile'].filename
        save_file_from_byte(buffer_data, filePath)  # 保存二进制文件
        text, ret = read_file(filePath)  # 读取二进制文件文本内容(ret=False表示文本解析异常)


    # file.save('D:\code\jms_v5\static\metadata')
    return 'file uploaded successfully'

if __name__ == '__main__':
    app.run()
