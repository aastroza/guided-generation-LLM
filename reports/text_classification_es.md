# Introduccion a la Generación Guiada de Texto usando LLMs.

> [!NOTE]
> Este post está inspirado en [esta publicación de LinkedIn de Luis Herreros](https://www.linkedin.com/feed/update/urn:li:activity:7137413404217991168/).


> [!IMPORTANT]
> Todo el código desarrollado para escribir este post puedes encontrarlos en este [Jupyter Notebook](../notebooks/text_classification.ipynb).

## Datos

```json
{"input": [{"role": "system",
   "content": "Respond with only a 1 or 0 to signify if the user's message includes sarcasm, or not"},
  {"role": "user",
   "content": "thirtysomething scientists unveil doomsday clock of hair loss"}],
 "ideal": "1"}
```

## Generación no Guiada

```python
from openai import OpenAI

client = OpenAI()

def generate_response(sample: dict, model: str):
    response = client.chat.completions.create(
        model=model,
        messages=sample['input'],
        temperature=0.0
    )
    return response.choices[0].message.content
```

## Generación Guiada

```python
import enum
from pydantic import BaseModel

class Labels(str, enum.Enum):
    """Enumeration for single-label text classification."""
    SARCASM = "sarcasm"
    NOT_SARCASM = "not_sarcasm"

class SinglePrediction(BaseModel):
    """
    Class for a single class label prediction.
    """
    class_label: Labels
```

```python
from openai import OpenAI
import instructor

client = instructor.patch(OpenAI())

def classify(sentence: str, model: str):
    """Perform single-label classification on the input text."""
    response = client.chat.completions.create(
        model=model,
        response_model=SinglePrediction,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"Classify the following text: {sentence}",
            },
        ],
    )

    response_message = '1' if response.class_label == Labels.SARCASM else '0'
    return response_message
```
## Resultados

El resumen de los experimentos realizados es el siguiente:

<img src="figures/sarcasm_accuracy_vs_cost.png" alt="sarcasm" width="600"/>

Cosas que se pueden inferir del gráfico:

- Las versiones guiadas de tanto GPT-3.5 como GPT-4 tienen mayor precisión que sus contrapartes no guiadas. En particular la ganancia de guiar a GPT-3.5 es bastante considerable, pasando de un 54.6% a un 64.1% de acierto sin aumentar demasiado los costos.
- GPT-4, ya sea guiado o no guiado, tiene mayor precisión que GPT-3.5 a cualquier costo dado.
- El intento de replicar la [idea de Luis del mix de modelos](https://www.linkedin.com/feed/update/urn:li:activity:7137413404217991168/) no salió tan bien en este caso. Probablemente haya que darle una revisión más profunda.
- El costo parece aumentar linealmente con la precisión, indicando que los modelos más precisos son más caros de utilizar. Este tipo de análisis es útil para determinar los compromisos entre la precisión y el costo, usar LLMs es caro y por supuesto que el gasto es un factor en discusión al construir productos.

## Ideas futuras

Por supuesto que para realizar mejores conclusiones deberíamos realizar experimentos sobre otros conjuntos de datos, usar distintas estrategias de prompting, explorar los otros modos de funcionamiento de [Instructor](https://github.com/jxnl/instructor), incluir otros tipos de tareas más complejas (por ejemplo la generación de grafos de conocimiento es algo que me interesa mucho) y comparar distintos frameworks (estoy muy entusiasmado por comprar resultados usando [Outlines](https://github.com/outlines-dev/outlines)). Es un trabajo que me entusiasma así que con el paso del tiempo espero construir un verdadero benchmark de generación guiada. 

Así que seguiré profundizando en el tema en futuros posts. Cualquier sugerencia es bienvenida.