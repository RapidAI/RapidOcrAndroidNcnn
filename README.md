# RapidOcrAndroidNcnn

[![Issue](https://img.shields.io/github/issues/RapidAI/RapidOcrAndroidNcnn.svg)](https://github.com/RapidAI/RapidOcrAndroidNcnn/issues)
[![Star](https://img.shields.io/github/stars/RapidAI/RapidOcrAndroidNcnn.svg)](https://github.com/RapidAI/RapidOcrAndroidNcnn)

<details open>
    <summary>目录</summary>

- [RapidOcrAndroidNcnn](#RapidOcrAndroidNcnn)
    - [联系方式](#联系方式)
    - [项目完整源码](#项目完整源码)
    - [APK下载](#APK下载)
    - [简介](#简介)
    - [总体说明](#总体说明)
    - [更新说明](#更新说明)
    - [编译说明](#编译说明)
    - [使用说明](#使用说明)
    - [项目结构](#项目结构)
    - [常见问题](#常见问题)
        - [输入参数说明](#输入参数说明)
    - [关于作者](#关于作者)
    - [版权声明](#版权声明)
    - [示例图](#示例图)
        - [IMEI识别](#IMEI识别)
        - [身份证识别](#身份证识别)
        - [车牌识别](#车牌识别)

</details>

## 联系方式

* QQ群号：887298230(已满)，2群(755960114)

## 项目完整源码

* 整合好源码和依赖库的完整工程项目，可到Q群共享内下载或Release下载，以Project开头的压缩包文件为源码工程，例：Project_RapidOcrAndroidNcnn-版本号.7z
* 如果想自己折腾，则请继续阅读本说明

## APK下载

* 编译好的demo apk，可以在release中下载，或者Q群共享内下载，文件名例：RapidOcrAndroidNcnn-版本号-cpu-release.apk

## 简介

RapidOcr ncnn推理 for Android

采用ncnn神经网络前向计算框架[https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)

## 总体说明

1. 封装为独立的Library，可以编译为aar，作为模块来调用；
2. Native层以C++编写；
3. Demo App以Kotlin-JVM编写；
4. Android版与其它版本不同，包含了几个应用场景，包括相册识别、摄像头识别、手机IMEI号识别、摄像头身份证识别这几个功能页面；
5. 可选择CPU版或GPU版；CPU版仅支持CPU运算，最低支持API21，且安装包体积小；GPU版支持vulkan(GPU加速)，最低支持API24，安装包体积较大；

## 更新说明

#### 2022-10-21 update 1.0.0
* 初始版本

#### 2022-02-16 update 1.1.0

* 增加相册识别和相机识别停止按钮
* 添加 Java demo
* 适配ncnn 20221128

### [编译说明](./BUILD.md)

### [使用说明](./INSTRUCTIONS.md)

## 项目结构

```
RapidOcrAndroidNcnn
    ├── app               # demo app
    ├── capture           # 截图
    ├── common-aar        # app引用的aar库
    ├── keystore          # app签名密钥文件
    ├── OcrLibrary        # Ocr引擎库，包含Jni和C++代码
    └── scripts           # 编译脚本
```

## 常见问题

### 输入参数说明

请参考[Cpp项目说明](https://github.com/RapidAI/RapidOcrNcnn)

## 关于作者

* Android demo编写：[benjaminwan](https://github.com/benjaminwan)
* 模型来自：[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 版权声明

- OCR模型版权归[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)所有；
- 其它工程代码版权归本仓库所有者所有；

## 示例图

#### IMEI识别

![avatar](capture/detect_IMEI.gif)

#### 身份证识别

![avatar](capture/detect_id_card.gif)

#### 车牌识别

![avatar](capture/detect_plate.gif)

