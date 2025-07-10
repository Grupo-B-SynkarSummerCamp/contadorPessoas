# ---------------------------------------------------------------------------
# PASSO 1: IMAGEM DE BASE
# Especifique a imagem oficial do ROS que você está usando.
# Exemplo: ros:noetic-robot-focal, ros:melodic-desktop, etc.
# IMPORTANTE: Substitua 'ros:noetic-robot' pela tag exata que você usa.
# ---------------------------------------------------------------------------
FROM osrf/ros:noetic-desktop-full

# ---------------------------------------------------------------------------
# PASSO 2: CONFIGURAR O AMBIENTE E INSTALAR DEPENDÊNCIAS
# É uma boa prática definir o shell padrão e atualizar os pacotes.
# ---------------------------------------------------------------------------
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    # Adicione aqui outras dependências do sistema que você precisar
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# PASSO 3: INSTALAR BIBLIOTECAS PYTHON (YOLOv8 e outras)
# Instala as bibliotecas Python necessárias para o seu script de detecção.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    opencv-python \
    ultralytics \
    supervision

# ---------------------------------------------------------------------------
# PASSO 4: CRIAR E CONFIGURAR A ÁREA DE TRABALHO (WORKSPACE) CATKIN
# ---------------------------------------------------------------------------
# Cria o diretório da workspace dentro da imagem
RUN mkdir -p /root/catkin_ws/src

# Define o diretório de trabalho para os comandos seguintes
WORKDIR /root/catkin_ws

# Copia o seu pacote (que está no seu computador) para dentro da imagem
# Assumimos que seu pacote 'contador_pessoas' está em 'src/contador_pessoas'
# relativo a este Dockerfile.
COPY ./src /root/catkin_ws/src

# Compila a workspace
RUN . /opt/ros/noetic/setup.bash && catkin_make

# ---------------------------------------------------------------------------
# PASSO 5: CONFIGURAR O AMBIENTE DE ENTRADA (BASHRC)
# Este comando faz com que o ambiente do ROS seja carregado automaticamente
# toda vez que você iniciar um terminal no contêiner.
# ---------------------------------------------------------------------------
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
