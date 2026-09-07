# Sample Corpus Document

Replace this file with your own documents (e.g. your NRSC/ISRO internship report,
project READMEs, or design case studies). One markdown or text file per document,
or one big file — the chunker in `src/rag/index.py` will split it either way.

## About Retrieval-Augmented Generation

Retrieval-Augmented Generation combines a retriever, which finds relevant text
chunks from a knowledge base, with a language model, which generates an answer
conditioned on those chunks. This grounds the model's output in real source
material rather than relying purely on parametric memory, which reduces
hallucination and allows the system to answer questions about content the
underlying model was never trained on.

## Why Evaluation Matters

A RAG system can fail in two independent places: the retriever can fail to find
the right context, or the generator can fail to use the context it was given
faithfully. Evaluating the two stages separately is the only way to know which
part of the pipeline needs improvement when answer quality is poor.


# ADAS Sensors

Advanced Driver Assistance Systems (ADAS) rely on a combination of sensors to
perceive the vehicle's surroundings. Radar sensors detect the distance and
speed of nearby objects and work reliably in poor weather. LiDAR uses laser
pulses to build a precise 3D map of the environment. Ultrasonic sensors are
used for short-range detection, such as parking assistance. Cameras provide
visual recognition of lane markings, traffic signs, and pedestrians. Most
production ADAS systems fuse data from several of these sensors together to
improve reliability, since each sensor type has different strengths and
blind spots.