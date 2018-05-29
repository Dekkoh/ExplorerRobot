# Explorer Robot - Inteligencia Artificial

Com base numa pesquisa realizada no FTI sobre inteligencia artificial, foi proposto um projeto cuja sua funcão seria implantar uma IA dentro de um robo, para que o mesmo se locomovesse pelo espaço da Hello World, explorando os lugares com um único sensor de presença ultra-sônico, sem que ele bata nas paredes e ou móveis presentes.
Utilizamos a biblioteca TensorFlow para desenvolver o aprendizado do Robo com a técnica **_Reinforcement Learning_** que tem por objetivo fazer com que a IA cumpra algumas metas oferecendo recompensar no caso de um movimento correto e também punições quando o objetivo deixa de ser atingido.

## Pré-Requisitos

- Raspberry pi 3
- Mindstorm NXT
- Sensor Ultra-Sônico

## Começando ...

### Configurações

Este projeto foi desenvolvido em Pyhton em conjunto com algumas de suas bibliotecas então seguem as instalações necessárias:

1. Instale o pyhton3 e o pip3 no Raspberry

    ```
    sudo apt-get update
    sudo apt-get install python3
    sudo apt-get install python3-pip python3-dev
    ``` 

2. Instale as bibliotecas: numpy e matplotlib no Raspberry

    **Responsável por gerar uma matriz multidimensional:**
    ```
    pip3 install numpy
    ```

    **Responsável por plotar gráficos com os dados gerados**
    ```
    pip3 install matplotlib
    ```

3. Instale a biblioteca do TensorFlow para implementar o aprendizado por reforço no Raspberry

    **Faça o download da biblioteca pelo link**
    ```
    wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
    ```
    
    **Altere o nome do arquivo para a versão mais recente antes de instalar com o comando:**
    ```
    mv tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl
    ```

    **Rode o comando abaixo para instalar:**
    ```
    sudo pip3 install tensorflow-1.1.0-cp35-cp35m-linux_armv7l.whl
    ```

## Rede Neural

> É nessessário um breve conhecimento sobre `Reinforcement Learn` para a realização deste projeto, com base nisso seguem alguns links com o intuito de clarear um pouco sobre este universo, caso não conheça nada sobre o mundo de Inteligência Artificial:

1. [https://www.oreilly.com/ideas/reinforcement-learning-with-tensorflow]

2. [https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0]

3. [http://conteudo.icmc.usp.br/pessoas/andre/research/neural/]

4. [http://neuralnetworksanddeeplearning.com/chap1.html]

Se seu conhecimento sobre o assunto seja suficiente, vamos para o que interessa.

Para este projeto foram utilizadas três camadas na rede neural, sendo elas: 
 - Camada de Input(entrada)
 - Camada intermediária, que contém os neurônios de processamento da rede
 - Camada de Output(Saída)   

Como citado anteriormente foi utilizado o treinamento por reforço neste projeto, onde em um determinado tempo t, o agente que está em um estado s, executa uma ação a, vai para um estado s, recebe uma recompensa r, que resumindo quer dizer que a cada iteração com o ambiente o nosso robo tende a mudar seu estado recebendo recompensas positivas e ou negativas dependendo de sua escolha. 

Por fim, utilizamos também para efeito de correção dos pesos da nossa rede o modo `Batch` que por sua vez faz uma única correção por ciclo, gerando o erro médio para que as correções sejam feitas.

> Este ciclo de treinamento é repetido até que se tenha um resultado satisfatório com base em métricas definidas via código.


## Programa Principal (Robo)

> Verifique se o Mindstorm está ligado e com o bluetooth ativado

**Fique atento a algumas partes importantes do código antes de rodar**

```
m_left = Motor(b, PORT_B)
m_right = Motor(b, PORT_A)
```

As constantes ``PORT_A`` e ``PORT_B`` se referem às portas onde os motores ``A`` e ``B`` devem ser respectivamente ligados para o programa funcionar em perfeito estado.
Assim como a linha:
```
distance = Ultrasonic(b, PORT_4,check_compatible=False).get_sample()
```
A constante ``PORT_4`` se refere a porta onde deve ser ligado o sensor ultra-sônico. Tendo cuidado com essas coisas o resto é somente a logica para o movimento do robo.

**Faça um conexão bluetooth com o Mindstorm**

Para se conectar ao Mindstorm via bluetooth é necessário instalar a biblioteca ``bluetoothctl`` com o comando:

```
sudo apt-get install pi-bluetooth
```

Após a instalação estar concluída é necessário ativar o bluetooth, buscar o dispositivo do Mindstorm e parear com o mesmo

1. Ativação do serviço
    ```
    sudo bluetoothctl
    ```

2. Habilitando para busca
    ```
    agent on
    ```

3. Inicializando como padrão
    ```
    default-agent
    ```

4. Busca de dispositivos próximos
    ```
    scan on
    ```

5. Ao encontrar um dispositivo digite o comando ``scan off`` para parar a busca e copie o endereço e execute o comando abaixo substituindo ``xx:xx:xx:xx:xx`` pelo endereço
    ```
    pair xx:xx:xx:xx:xx
    ```

> Fique atento pois irá aparecer uma solicitação no visor do Mindstorm com uma senha, assim que for confirmado a mesma senha deverá ser digitada no Raspberry quando for solicitada.

Após a conexão ser bem sucedida execute o seguinte comando na pasta raiz do projeto:
```
python3 main.py
```

## Desenvolvedores

* **Andre Tsuyoshi** - *Developer* - [email](andre.sakiyama@venturus.org.br)
* **Walter Inácio** - *Developer* - [email](walter.inacio@venturus.org.br)