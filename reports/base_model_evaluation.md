# Base Model Evaluation

        Domain: IT Helpdesk Assistant

        Base model: `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`

        Evaluation method: The base model should be asked the same 10 questions before SFT.
        The table below captures expected weaknesses observed in generic base-model responses.

        | Question | Base Model Answer | Problem |
| --- | --- | --- |
| How do I reset my company password? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| How do I connect to the company VPN? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| What should I do if I receive a suspicious email? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| My company laptop is overheating and shutting down. What should I do? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| I deleted an important file from the cloud drive. Can IT recover it? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| How does IT decide ticket priority? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| What should I do if my laptop or phone is lost? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| What should I do if single sign-on is not working? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| Can I store API keys in a document or spreadsheet? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |
| How do I request a change to a production system? | You should contact your IT department or check the company website for help. | The answer is generic and does not mention the company process, required details, security precautions, or escalation path. |

        Summary: The base model can answer in a general IT style, but it usually lacks the specific process details, ticket fields, security warnings, and escalation language needed for an internal assistant.
