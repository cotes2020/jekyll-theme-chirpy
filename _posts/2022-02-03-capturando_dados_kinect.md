---
title: Aquisição de dados usando o Sensor Kinect do Xbox 360
author:
  name: Natalia Amorim
  link: https://github.com/NataliaCarvalho03
date: 2022-02-03 21:14:34 -0300
categories: [kinect, openkinect, c++]
tags: [kinect libfreenec c++]
pin: false
---

Neste artigo trago um rápido tutorial para que você possa começar a utilizar o sensor kinect do xbox 360 (Kinect v1) para adquirir imagens RGB e mapa de profundidade. Usaremos a libfreenect (OpenKinect) e a OpenCV em linguagem C++ para aprendermos a obter dados do kinect.

# 1. Instalando as bibliotecas

## Compilando a OpenCV

```bash
#dependencias da opencv
sudo apt-get update
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran libeigen3-dev

cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd ~/opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D INSTALL_PYTHON_EXAMPLES=OFF \
 -D INSTALL_C_EXAMPLES=ON \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
 -D BUILD_EXAMPLES=ON ..
make -j4
sudo make install
```

## Compilando a libfreenect

```bash
cd ~
sudo apt-get install git build-essential libusb-1.0-0-dev cython3 libglfw3-dev

#instalando a librealsense
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

sudo apt-get install librealsense2-dkms

sudo apt-get install librealsense2-utils

# para compilar os exemplos
sudo apt-get install freeglut3-dev libxmu-dev libxi-dev

git clone https://github.com/OpenKinect/libfreenect
cd libfreenect
mkdir build && cd build

cmake -DBUILD_PYTHON3=ON -DCYTHON_EXECUTABLE=/usr/bin/cython3 ..
#pra caso você queira utilizar também com python
#lembre-se que você deve ter o python e a numpy instalados para usar as flags acima
make
sudo make install
sudo ldconfig -v

```

# 2. Criando o Projeto

## Preparando o CMakeLists

Crie uma pasta chamada kinect-project em algum lugar de sua preferência. Dentro desta pasta, crie um arquivo chamado main.cpp e um arquivo chamado CMakeLists.txt.

A estrutura de seu projeto deve ficar desta forma:

```
--kinect-project
         |__ main.cpp
         |__ CMakeLists.txt
```

Dentro do arquivo CMakeLists.txt coloque o conteúdo abaixo:

```cmake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(libfreenect REQUIRED)


include_directories( /usr/local/include/libfreenect/ #verifique se os includes da sua instalação realmente estão neste caminho
                     /usr/local/include/libusb-1.0/  #idem pra esses includes aqui
)

add_executable(
    kinect_project main.cpp
)

target_link_libraries(kinect_project  ${OpenCV_LIBS}
                                      ${CMAKE_THREAD_LIBS_INIT}
                                      ${FREENECT_LIBRARIES} freenect
)

```

## Escrevendo o Código

Caso esteja com pressa, o código-fonte completo deste artigo pode ser obtido [aqui](https://github.com/NataliaCarvalho03/kinect-pointCloud/blob/main/Calibration/kinect_calibration.cpp). (Apesar do módulo se chamar Calibration, o código ainda não faz isso).

### Inicializando a freenect e acessando o dispositivo

A primeira coisa que precisamos fazer é adicionar os includes necessários para o nosso projeto:

```c++
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "libfreenect.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
```

Agora podemos começar a escrever o conteúdo da nossa função main. A primeira coisa a se fazer é **inicializar a freenect**:

```c++
int main(int argc, char** argv)
{
    freenect_context* fn_ctx;
	int ret = freenect_init(&fn_ctx, NULL);
	if (ret < 0)
		return ret; //Caso não tenha sucesso em inicializar, para a execução
}
```

O próximo passo é **encontrar os dispositivos kinect conectados em sua máquina** (você conectou o kinect, não é?). Para isso, adicione o seguinte trecho no seu código:

```c++
int num_devices = ret = freenect_num_devices(fn_ctx);
if (ret < 0)
    return ret;
if (num_devices == 0)
{
    std::cout << "Nenhum dispositivo conectado!" << std::endl;
    freenect_shutdown(fn_ctx);
    return 1;
}
```

Depois que encontramos os dispositivos conectados, devemos acessar pelo menos um deles para começarmos a obter os dados. Fazemos isso adicionando o seguinte trecho de código:

```c++
freenect_device* fn_dev; // Variável que vai guardar as informações do dispositivo
ret = freenect_open_device(fn_ctx, &fn_dev, 0); //Acessando o primeiro dispositivo que possui indice 0 na lista
if (ret < 0)
{
    freenect_shutdown(fn_ctx); // Caso não consiga acessar o dispositivo, encerra a freenect e para a execução
    return ret;
}
```

Agora que já temos acesso ao dispositivo, podemos configurar o modo que receberemos as informações do sensor RGB e do sensor de profundidade:

```c++
ret = freenect_set_depth_mode(fn_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM));
if (ret < 0)
{
    freenect_shutdown(fn_ctx);
    return ret;
}
ret = freenect_set_video_mode(fn_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
if (ret < 0)
{
    freenect_shutdown(fn_ctx);
    return ret;
}
```

### Criando as funções de Callback

Para para sermos capazes de manipular os frames que recebemos dos sensores RGB e IR, precisamos definir as funções de callback, que serão chamadas sempre que um novo frame chegar. Então, vamos criar uma função callback para receber o frame RGB e outra callback para receber o frame de profundidade.

```c++
//função callback para o frame de profundidade

void depth_cb(freenect_device* dev, void* data, uint32_t timestamp)
{
	printf("Received depth frame at %d\n", timestamp);
	cv::Mat depthMat(cv::Size(640,480),CV_16UC1);
	uint16_t* depth = static_cast<uint16_t*>(data);
	depthMat.data = (uchar*) data;
	cv::imwrite("depth.png", depthMat);
}

//Função callback para o frame RGB
void video_cb(freenect_device* dev, void* data, uint32_t timestamp)
{
	printf("Received video frame at %d\n", timestamp);
	cv::Mat rgbMat(cv::Size(640,480), CV_8UC3, cv::Scalar(0));
	uint8_t* rgb = static_cast<uint8_t*>(data);
	rgbMat.data = rgb;
	cv::imwrite("rgb.png", rgbMat);
}
```

Agora, dentro da função main precisamos precisamos "registrar" essas funções callback para que sejam chamadas a cada novo frame:

```c++
freenect_set_depth_callback(fn_dev, depth_cb); //Lembra que fn_dev é a variável que guarda as informações do dispositivo, né?
freenect_set_video_callback(fn_dev, video_cb);
```

### Iniciando a captura de frames

Para iniciar a captura dos frames, utilizamos a função ```freenect_start_depth``` para capturar os frames de profundidade e ```freenect_start_video``` para captura dos frames RGB. Copie e cole o trecho abaixo:

```c++
ret = freenect_start_depth(fn_dev);
if (ret < 0)
{
    freenect_shutdown(fn_ctx);
    return ret;
}
ret = freenect_start_video(fn_dev);
if (ret < 0)
{
    freenect_shutdown(fn_ctx);
    return ret;
}
```

Para manter a captura rodando até que uma interrupção seja requisitada, vamos usar um recurso chamado signals (sinais). A ideia é emitir um sinal toda vez que uma requisição de interrupção do nosso programa for feita. Para isso, vamos declarar uma variável global que armazena o estado do nosso programa e uma função que irá lidar com o sinal (handler). Copie e cole o trecho de código abaixo:

```c++
//variável global
volatile bool running = true;

//função que será chamada quando o sinal de interrupção for emitido
void signalHandler(int signal)
{
	if (signal == SIGINT
	 || signal == SIGTERM
	 || signal == SIGQUIT)
	{
		running = false;
	}
}
```

Agora, dentro da função main colocamos o seguinte loop:

```c++
while (running && freenect_process_events(fn_ctx) >= 0)
{

}
```

Depois que o nosso loop for interrompido, devemos parar a execução da freenect de forma apropriada. Então, depois do loop, cole o seguinte trecho de código:

```c++
std::cout << "Parando a execução" << std::endl;

// Stop everything and shutdown.
freenect_stop_depth(fn_dev);
freenect_stop_video(fn_dev);
freenect_close_device(fn_dev);
freenect_shutdown(fn_ctx);

std::cout << "Encerrado com sucesso!" << std::endl;

return 0;
```

# Compilando e Executando o Projeto

Agora podemos finalmente compilar e executar nosso projeto. Para isso, utilizaremos o cmake, então basta executar os seguintes comandos dentro da pasta do nosso projeto:

```bash
mkdir build && cd build

cmake ..
make

./kinect_project
```

Pronto! Agora você verá alguns prints passando no seu terminal (prints que vieram das funções callback) e você pode requisitar a interrupção do programa com um Ctrl+c. Depois que a execução parar, é só olhar dentro da sua pasta build, as imagens rgb.png e depth.png estão salvas lá dentro.

Assim você acabou de adquirir uma imagem RGB e uma imagem de profundidade usando o sensor Kinect com a lifreenect e a OpenCV :)



Com muito carinho e um pouco de agressão...


[Natália Amorim](https://www.linkedin.com/in/natalia-carvalho-02901798/)

Engenheira em Visão Computacional e fundadora do Grupo OpenCV Brasil.