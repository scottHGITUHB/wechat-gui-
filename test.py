import pyautogui
import cv2
import os

def find_image_on_screen(image_path):
    # 截取屏幕图片
    pyautogui.screenshot(imageFilename="screen.bmp")

    # 读取屏幕截图和目标图片
    screenpic = cv2.imread("screen.bmp")
    mypic = cv2.imread(image_path)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测和计算图片的关键点和描述符
    screenPicKP, screenPicDES = sift.detectAndCompute(screenpic, None)
    myPicKP, myPicDES = sift.detectAndCompute(mypic, None)

    # 设置FLANN匹配参数
    trees = 100
    checks = 1000
    indexParams = dict(algorithm=0, trees=int(trees))
    searchParams = dict(checks=int(checks))

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    # 进行特征匹配
    matches = flann.knnMatch(screenPicDES, myPicDES, k=2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x[0].distance)

    # 寻找最佳匹配的位置
    x, y = None, None
    max_init_num = 0.4
    init_num = 0.1
    while init_num <= max_init_num:
        goodMatches = [m for m, n in matches if m.distance < init_num * n.distance]
        if goodMatches:
            index = int(len(goodMatches) / 2)
            try:
                x, y = screenPicKP[goodMatches[index].queryIdx].pt
                break
            except IndexError:
                pass
        init_num += 0.1

    # 清理临时文件
    os.remove("screen.bmp")

    # 返回结果
    return x, y
#
# if __name__ == "__main__":
#     # 测试代码
#     image_path = "seek.bmp"
#     x,y=find_image_on_screen(image_path)
#     pyautogui.moveTo(x, y, duration=1)
