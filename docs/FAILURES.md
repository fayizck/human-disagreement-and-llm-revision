# Collection problems and resolutions

This file describes problems encountered during data collection and how they were handled. None of these changes altered the selected examples, prompts, model settings, or analysis plan. Responses that had already been collected were not generated again.

## Study 1

### Startup import failure

An early run stopped before sending any requests because Python imported the wrong local `engine.py` file and could not find the required `reconcile` function. The import was changed to use the full package path, and a startup test was added before collection resumed. This failed run produced no model responses.

### Request failures

The main collection required 33,000 responses and made 32,994 requests. One OpenAI request and one Gemini request failed for technical reasons. Each request was tried once more with the same content and settings. These retries filled the original two positions and did not add examples to the study.

### Invalid initial classifications

Two OpenAI initial responses did not match one of the required NLI labels. They were kept as invalid responses and were not replaced. Because each initial answer was meant to lead to four follow-up questions, those eight follow-ups could not be asked. The final dataset therefore contains 32,992 responses, with all 33,000 planned positions accounted for.

### Quota-window pause

The final OpenAI portion paused when the daily request limit was reached. The collected responses were checked before the run continued from the next unanswered position. The earlier 200-example portion remained unchanged, and no response was recorded twice.

## Study 2

### Model-information check

The first run stopped before collection with `KeyError: 'kind'`. The error message tried to read a field that was present only when the model-information check failed. The message handling was corrected, while an actual model mismatch still stops the program. A regression test was added, and the database still contained no requests or responses after this failure.

### Invalid scientific outputs

Twelve initial answers did not match either required VitaminC label. Their 24 follow-up questions could not be asked. One neutral follow-up and one directed follow-up also returned invalid labels. These responses were kept and were not replaced.

### Final accounting

Study 2 contains 1,776 responses, and the remaining 24 planned follow-ups were not applicable because of the invalid initial answers described above. No response was recorded twice, and no planned position remains unexplained.

## Rule used during recovery

A failed request was repeated only with the same example, prompt, and model settings. Invalid model answers were kept as study results rather than replaced with more favorable answers.
