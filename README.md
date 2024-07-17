# chatbot
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NeuralChat is a customizable chat framework designed to create user own chatbot within few minutes on multiple architectures. This notebook is used to demonstrate how to build a talking chatbot on 4th Generation of Intel® Xeon® Scalable Processors Sapphire Rapids.\n",
    "\n",
    "The 4th Generation of Intel® Xeon® Scalable processor provides two instruction sets viz. AMX_BF16 and AMX_INT8 which provides acceleration for bfloat16 and int8 operations respectively."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Prepare Environment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Install intel extension for transformers:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Defaulting to user installation because normal site-packages is not writeable\n",
      "Requirement already satisfied: intel-extension-for-transformers in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (1.4.2)\n",
      "Requirement already satisfied: packaging in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from intel-extension-for-transformers) (23.2)\n",
      "Requirement already satisfied: numpy in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from intel-extension-for-transformers) (1.23.5)\n",
      "Requirement already satisfied: schema in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from intel-extension-for-transformers) (0.7.7)\n",
      "Requirement already satisfied: pyyaml in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from intel-extension-for-transformers) (6.0.1)\n",
      "Requirement already satisfied: neural-compressor in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from intel-extension-for-transformers) (2.6)\n",
      "Requirement already satisfied: transformers in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from intel-extension-for-transformers) (4.41.2)\n",
      "Requirement already satisfied: deprecated>=1.2.13 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (1.2.14)\n",
      "Requirement already satisfied: opencv-python-headless in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (4.10.0.84)\n",
      "Requirement already satisfied: pandas in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (2.2.2)\n",
      "Requirement already satisfied: Pillow in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (10.2.0)\n",
      "Requirement already satisfied: prettytable in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (3.10.0)\n",
      "Requirement already satisfied: psutil in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (5.9.8)\n",
      "Requirement already satisfied: py-cpuinfo in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (9.0.0)\n",
      "Requirement already satisfied: requests in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (2.32.3)\n",
      "Requirement already satisfied: scikit-learn in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (1.4.0)\n",
      "Requirement already satisfied: pycocotools in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from neural-compressor->intel-extension-for-transformers) (2.0.8)\n",
      "Requirement already satisfied: filelock in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from transformers->intel-extension-for-transformers) (3.15.4)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.23.0 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from transformers->intel-extension-for-transformers) (0.23.4)\n",
      "Requirement already satisfied: regex!=2019.12.17 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from transformers->intel-extension-for-transformers) (2024.5.15)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from transformers->intel-extension-for-transformers) (0.19.1)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from transformers->intel-extension-for-transformers) (0.4.3)\n",
      "Requirement already satisfied: tqdm>=4.27 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from transformers->intel-extension-for-transformers) (4.66.4)\n",
      "Requirement already satisfied: wrapt<2,>=1.10 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from deprecated>=1.2.13->neural-compressor->intel-extension-for-transformers) (1.16.0)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in /home/uc310dfd2c0fb9a63dfa008d96b796af/.local/lib/python3.9/site-packages (from huggingface-hub<1.0,>=0.23.0->transformers->intel-extension-for-transformers) (2024.5.0)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from huggingface-hub<1.0,>=0.23.0->transformers->intel-extension-for-transformers) (4.9.0)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from pandas->neural-compressor->intel-extension-for-transformers) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from pandas->neural-compressor->intel-extension-for-transformers) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from pandas->neural-compressor->intel-extension-for-transformers) (2023.4)\n",
      "Requirement already satisfied: wcwidth in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from prettytable->neural-compressor->intel-extension-for-transformers) (0.2.13)\n",
      "Requirement already satisfied: matplotlib>=2.1.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from pycocotools->neural-compressor->intel-extension-for-transformers) (3.8.2)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from requests->neural-compressor->intel-extension-for-transformers) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from requests->neural-compressor->intel-extension-for-transformers) (3.6)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from requests->neural-compressor->intel-extension-for-transformers) (2.2.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from requests->neural-compressor->intel-extension-for-transformers) (2024.2.2)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from scikit-learn->neural-compressor->intel-extension-for-transformers) (1.10.1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from scikit-learn->neural-compressor->intel-extension-for-transformers) (1.3.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from scikit-learn->neural-compressor->intel-extension-for-transformers) (3.3.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (0.12.1)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (4.47.2)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (1.4.5)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (3.1.1)\n",
      "Requirement already satisfied: importlib-resources>=3.2.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (6.1.1)\n",
      "Requirement already satisfied: six>=1.5 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from python-dateutil>=2.8.2->pandas->neural-compressor->intel-extension-for-transformers) (1.16.0)\n",
      "Requirement already satisfied: zipp>=3.1.0 in /opt/intel/oneapi/intelpython/lib/python3.9/site-packages (from importlib-resources>=3.2.0->matplotlib>=2.1.0->pycocotools->neural-compressor->intel-extension-for-transformers) (3.17.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install intel-extension-for-transformers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Install Requirements:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/intel/intel-extension-for-transformers.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/\n",
    "!pip install -r requirements_cpu.txt\n",
    "%cd ../../../"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Build your chatbot 💻"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Chat"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Giving NeuralChat the textual instruction, it will respond with the textual response."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Chat With Retrieval Plugin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading model Intel/neural-chat-7b-v3-1\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b50f465c0b0e4a2b8a9c019a1afbcc61",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Suzuki is a renowned Japanese automobile manufacturer founded in 1909 by Michio Suzuki. Initially known for producing weaving looms, they later expanded into motorcycles and eventually cars. Today, Suzuki is recognized globally for its innovative designs, fuel efficiency, and commitment to quality. Their product lineup includes a variety of vehicles such as small cars, SUVs, and motorcycles. The company has also made significant contributions to motorsports, with their racing heritage dating back to the early days of motorcycle racing. In summary, Suzuki is a versatile brand that has evolved over time, leaving a lasting impact on both the automotive and motorcycling industries. \n",
      "\n",
      "As the sun sets behind the horizon, casting a golden glow upon the cityscape, Suzuki's legacy shines brightly. From humble beginnings in weaving looms, they have transformed into a global powerhouse in the automotive world. Their passion for innovation and dedication to excellence continues to inspire generations of drivers and riders alike. As the wheels keep turning, so does the spirit of Suzuki – ever striving to create vehicles that embody the essence of freedom and adventure.\n"
     ]
    }
   ],
   "source": [
    "# BF16 Optimization\n",
    "from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig\n",
    "from intel_extension_for_transformers.transformers import MixedPrecisionConfig\n",
    "config = PipelineConfig(optimization_config=MixedPrecisionConfig())\n",
    "chatbot = build_chatbot(config)\n",
    "response = chatbot.predict(query=\"Tell me about Suzuki.\")\n",
    "print(response)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "User could also leverage NeuralChat Retrieval plugin to do domain specific chat by feding with some documents like below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/\n",
    "!pip install -r requirements.txt\n",
    "%cd ../../../../../../"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!mkdir docs\n",
    "%cd docs\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/docs/sample.jsonl\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/docs/sample.txt\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/docs/sample.xlsx\n",
    "%cd .."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from intel_extension_for_transformers.neural_chat import PipelineConfig\n",
    "from intel_extension_for_transformers.neural_chat import build_chatbot\n",
    "from intel_extension_for_transformers.neural_chat import plugins\n",
    "plugins.retrieval.enable=True\n",
    "plugins.retrieval.args[\"input_path\"]=\"./docs/\"\n",
    "config = PipelineConfig(plugins=plugins)\n",
    "chatbot = build_chatbot(config)\n",
    "response = chatbot.predict(\"How many cores does the Intel® Xeon® Platinum 8480+ Processor have in total?\")\n",
    "print(response)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Voice Chat with ASR & TTS Plugin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the context of voice chat, users have the option to engage in various modes: utilizing input audio and receiving output audio, employing input audio and receiving textual output, or providing input in textual form and receiving audio output.\n",
    "\n",
    "For the Python API code, users have the option to enable different voice chat modes by setting ASR and TTS plugins enable or disable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd ./intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/\n",
    "!pip install -r requirements.txt\n",
    "%cd ../../../../../../"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/speaker_embeddings/spk_embed_default.pt\n",
    "!curl -OL https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from intel_extension_for_transformers.neural_chat import PipelineConfig\n",
    "from intel_extension_for_transformers.neural_chat import build_chatbot\n",
    "from intel_extension_for_transformers.neural_chat import plugins\n",
    "plugins.tts.enable = True\n",
    "plugins.tts.args[\"output_audio_path\"] = \"./response.wav\"\n",
    "plugins.asr.enable = True\n",
    "\n",
    "config = PipelineConfig(plugins=plugins)\n",
    "chatbot = build_chatbot(config)\n",
    "result = chatbot.predict(query=\"./sample.wav\")\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Low Precision Optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## BF16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# BF16 Optimization\n",
    "from intel_extension_for_transformers.neural_chat.config import PipelineConfig\n",
    "from intel_extension_for_transformers.transformers import MixedPrecisionConfig\n",
    "config = PipelineConfig(optimization_config=MixedPrecisionConfig())\n",
    "chatbot = build_chatbot(config)\n",
    "response = chatbot.predict(query=\"Tell me about Intel Xeon Scalable Processors.\")\n",
    "print(response)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "neural-chat",
   "language": "python",
   "name": "neural-chat"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
