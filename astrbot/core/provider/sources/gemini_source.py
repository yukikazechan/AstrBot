import traceback
import base64
import json
import aiohttp
import asyncio # Added import for asyncio.TimeoutError
from astrbot.core.utils.io import download_image_by_url
from astrbot.core.db import BaseDatabase
from astrbot.api.provider import Provider, Personality
from astrbot import logger
from astrbot.core.provider.func_tool_manager import FuncCall
from typing import List
from ..register import register_provider_adapter
from astrbot.core.provider.entites import LLMResponse

class SimpleGoogleGenAIClient():
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        if api_base.endswith("/"):
            self.api_base = api_base[:-1]
        else:
            self.api_base = api_base

        self.client = aiohttp.ClientSession(trust_env=True)

    async def models_list(self) -> List[str]:
        request_url = f"{self.api_base}/v1beta/models?key={self.api_key}"
        async with self.client.get(request_url, timeout=10) as resp:
            # Added status check
            if resp.status != 200:
                try:
                    error_response = await resp.json()
                    logger.error(f"Gemini API error response: Status {resp.status}, Body: {error_response}")
                    if "error" in error_response and "message" in error_response["error"]:
                        raise Exception(f"Gemini API Error: {error_response['error']['message']} (Status: {resp.status})")
                    else:
                        raise Exception(f"Gemini API Error: Unexpected error format (Status: {resp.status}), Body: {error_response}")
                except aiohttp.ContentTypeError:
                    error_text = await resp.text()
                    logger.error(f"Gemini API error response: Status {resp.status}, Body: {error_text}")
                    raise Exception(f"Gemini API Error: Received non-JSON error response (Status: {resp.status}), Body: {error_text}")
                except Exception as e:
                    logger.error(f"Failed to parse Gemini API error response: {e}")
                    raise Exception(f"Gemini API Error: Failed to parse error response (Status: {resp.status})")

            response = await resp.json()
            models = []
            if "models" in response: # Added check for "models" key
                for model in response["models"]:
                    if 'supportedGenerationMethods' in model and 'generateContent' in model["supportedGenerationMethods"]: # Added check for keys
                        if "name" in model: # Added check for "name" key
                             models.append(model["name"].replace("models/", ""))
            else:
                logger.warning(f"Unexpected response format from models_list: {response}")
            return models


    async def generate_content(
        self,
        contents: List[dict],
        model: str="gemini-1.5-flash",
        system_instruction: str="",
        tools: dict=None
    ):
        payload = {}
        if system_instruction:
            payload["system_instruction"] = {
                "parts": [{"text": system_instruction}] # Ensure system instruction is in parts list
            }
        if tools:
            payload["tools"] = [tools]
        payload["contents"] = contents
        logger.debug(f"payload: {payload}")
        request_url = f"{self.api_base}/v1beta/models/{model}:generateContent?key={self.api_key}"
        async with self.client.post(request_url, json=payload, timeout=10) as resp:
            # Added status check
            if resp.status != 200:
                try:
                    error_response = await resp.json()
                    logger.error(f"Gemini API error response: Status {resp.status}, Body: {error_response}")
                    if "error" in error_response and "message" in error_response["error"]:
                         # Check if it's a timeout error from the infrastructure
                        if resp.status == 504 or (isinstance(error_response.get("error", {}).get("message"), str) and "timed out" in error_response["error"]["message"].lower()):
                             raise aiohttp.ClientTimeout(message=error_response["error"]["message"]) # Raise timeout error for retry logic
                        raise Exception(f"Gemini API Error: {error_response['error']['message']} (Status: {resp.status})")
                    else:
                        raise Exception(f"Gemini API Error: Unexpected error format (Status: {resp.status}), Body: {error_response}")
                except aiohttp.ContentTypeError:
                    error_text = await resp.text()
                    logger.error(f"Gemini API error response: Status {resp.status}, Body: {error_text}")
                    raise Exception(f"Gemini API Error: Received non-JSON error response (Status: {resp.status}), Body: {error_text}")
                except Exception as e:
                    # Re-raise timeout exceptions caught here
                    if isinstance(e, aiohttp.ClientTimeout) or isinstance(e, asyncio.TimeoutError):
                         raise e
                    logger.error(f"Failed to parse Gemini API error response: {e}")
                    raise Exception(f"Gemini API Error: Failed to parse error response (Status: {resp.status})")


            response = await resp.json()
            return response


@register_provider_adapter("googlegenai_chat_completion", "Google Gemini Chat Completion 提供商适配器")
class ProviderGoogleGenAI(Provider):
    def __init__(
        self,
        provider_config: dict,
        provider_settings: dict,
        db_helper: BaseDatabase,
        persistant_history = True,
        default_persona: Personality=None
    ) -> None:
        super().__init__(provider_config, provider_settings, persistant_history, db_helper, default_persona)
        self.chosen_api_key = None
        self.api_keys: List = provider_config.get("key", [])
        self.chosen_api_key = self.api_keys[0] if len(self.api_keys) > 0 else None

        self.client = SimpleGoogleGenAIClient(
            api_key=self.chosen_api_key,
            api_base=provider_config.get("api_base", None)
        )
        self.set_model(provider_config['model_config']['model'])

    async def get_human_readable_context(self, session_id, page, page_size):
        if session_id not in self.session_memory:
            raise Exception("会话 ID 不存在")
        contexts = []
        temp_contexts = []
        for record in self.session_memory[session_id]:
             # Added checks for role and content before accessing
            if isinstance(record, dict) and 'role' in record and 'content' in record:
                if record['role'] == "user":
                    # Ensure content is string for human readable context
                    if isinstance(record['content'], str):
                         temp_contexts.append(f"User: {record['content']}")
                    elif isinstance(record['content'], list):
                         # Handle multimodal content for display
                         text_parts = [part.get('text', '') for part in record['content'] if isinstance(part, dict) and part.get('type') == 'text']
                         temp_contexts.append(f"User: {''.join(text_parts)} (包含图片)")
                elif record['role'] == "assistant":
                    if isinstance(record['content'], str):
                         temp_contexts.append(f"Assistant: {record['content']}")
                # Assuming other roles like 'tool' are not needed in human readable context, or handle them if necessary
                if record['role'] in ["user", "assistant"]:
                    contexts.insert(0, temp_contexts)
                    temp_contexts = []
            else:
                 logger.warning(f"Skipping invalid record in session_memory: {record}")


        # 展平 contexts 列表
        # The logic here seems slightly off if temp_contexts is not empty at the end of the loop.
        # It might be intended to group user/assistant turns. Let's keep the original logic for now but note it.
        # Correct flattening might require a different approach depending on desired output structure.
        # For now, assuming the original logic's intent for human readability.
        flat_contexts = [item for sublist in contexts for item in sublist]
        # Add any remaining temp_contexts (last turn if loop ended on user)
        flat_contexts.extend(temp_contexts)


        # 计算分页
        total_items = len(flat_contexts)
        paged_contexts = flat_contexts[(page-1)*page_size:page*page_size]
        total_pages = total_items // page_size
        if total_items % page_size != 0:
            total_pages += 1

        return paged_contexts, total_pages

    async def get_models(self):
        return await self.client.models_list()

    async def pop_record(self, session_id: str, pop_system_prompt: bool = False):
        '''
        弹出第一条记录
        '''
        if session_id not in self.session_memory:
            raise Exception("会话 ID 不存在")

        if len(self.session_memory[session_id]) == 0:
            return None

        # Note: The original logic for popping system prompt seems complex and potentially buggy
        # if multiple system prompts exist or if the structure is not user/assistant turns.
        # A simpler approach might be needed depending on how system prompts are truly handled.
        # Keeping original logic but be aware of potential issues.

        record_to_pop = None
        pop_index = -1

        for i in range(len(self.session_memory[session_id])):
            record = self.session_memory[session_id][i]
            # Check if the record is a dict and has 'role' before accessing
            if isinstance(record, dict) and 'role' in record:
                 # If not popping system prompt, check if this is a system prompt
                if not pop_system_prompt and record.get('role') == "system":
                    # Check if there's any non-system message after this one
                    has_non_system_after = any(isinstance(r, dict) and r.get('role') != 'system' for r in self.session_memory[session_id][i+1:])
                    if not has_non_system_after:
                        # If only system prompts remain after this one, skip it
                        continue

                # If we reach here, this record is eligible to be popped
                record_to_pop = record
                pop_index = i
                break # Pop the first eligible record

            else:
                 # Log invalid record found in session_memory
                 logger.warning(f"Found invalid record format in session_memory, skipping for pop_record: {record}")
                 continue # Skip invalid record and continue search


        if pop_index != -1:
             return self.session_memory[session_id].pop(pop_index)

        return None # No eligible record found to pop


    async def _query(self, payloads: dict, tools: FuncCall) -> LLMResponse:
        tool = None
        if tools:
            tool = tools.get_func_desc_google_genai_style()
            # Removed the `if not tool: tool = None` as get_func_desc_google_genai_style
            # should return None if no tools are available or properly formatted.

        system_instruction = ""
        # Added checks for 'messages' key and ensure it's a list
        if "messages" in payloads and isinstance(payloads["messages"], list):
            for message in payloads["messages"]:
                 # Added check for 'role' and 'content' keys
                if isinstance(message, dict) and "role" in message and "content" in message:
                    if message["role"] == "system":
                        # Ensure system instruction is a string
                        if isinstance(message["content"], str):
                             system_instruction = message["content"]
                             break # Assuming only one system instruction at the beginning
                        else:
                             logger.warning(f"Skipping system message with non-string content: {message}")
                else:
                     logger.warning(f"Skipping invalid message format in payloads: {message}")
        else:
            logger.warning(f"Payloads missing 'messages' key or 'messages' is not a list: {payloads}")
            # Decide how to handle payloads without messages, perhaps raise an error or return early.
            # For now, proceeding with empty google_genai_conversation if no valid messages.


        google_genai_conversation = []
        # Re-iterate messages to build the conversation for Gemini (excluding system instruction which is handled separately)
        if "messages" in payloads and isinstance(payloads["messages"], list):
            for message in payloads["messages"]:
                 # Added check for 'role' and 'content' keys
                if isinstance(message, dict) and "role" in message and "content" in message:
                    # Skip system messages here as they are handled by system_instruction
                    if message["role"] == "system":
                        continue

                    if message["role"] == "user":
                        if isinstance(message["content"], str):
                            google_genai_conversation.append({
                                "role": "user",
                                "parts": [{"text": message["content"]}]
                            })
                        elif isinstance(message["content"], list):
                            # images and text parts
                            parts = []
                            # Added robust checking for parts in multimodal content
                            for part in message["content"]:
                                if not isinstance(part, dict):
                                    logger.warning(f"Skipping user message part as it is not a dictionary: {part}")
                                    continue

                                if part.get("type") == "text" and "text" in part:
                                    # Ensure text part content is a string
                                    if isinstance(part["text"], str):
                                        parts.append({"text": part["text"]})
                                    else:
                                         logger.warning(f"Skipping text part with non-string content: {part}")
                                elif part.get("type") == "image_url" and "image_url" in part and "url" in part["image_url"]:
                                     # Ensure image url is a string
                                    if isinstance(part["image_url"]["url"], str):
                                        # Added check for base64 prefix before replace
                                        image_data = part["image_url"]["url"]
                                        if image_data.startswith("data:image/jpeg;base64,"):
                                             image_data = image_data.replace("data:image/jpeg;base64,", "") # base64
                                        # You might want to add handling for other image types if necessary
                                        parts.append({"inline_data": {
                                            "mime_type": "image/jpeg", # Assuming JPEG, should be dynamic if possible
                                            "data": image_data
                                        }})
                                    else:
                                         logger.warning(f"Skipping image_url part with non-string url: {part}")
                                else:
                                     logger.warning(f"Skipping user message part due to unexpected format: {part}")

                            if parts: # Only add user message if there are valid parts
                                google_genai_conversation.append({
                                    "role": "user",
                                    "parts": parts
                                })
                            else:
                                 logger.warning(f"Skipping user message with content list as no valid parts were found after processing: {message}")
                        else:
                             logger.warning(f"Skipping user message with unexpected content type: {type(message['content'])}")


                    elif message["role"] == "assistant":
                        # Ensure assistant content is a string
                        if isinstance(message["content"], str):
                            google_genai_conversation.append({
                                "role": "model",
                                "parts": [{"text": message["content"]}]
                            })
                        else:
                             logger.warning(f"Skipping assistant message with non-string content: {message}")
                    # Add handling for 'tool' role messages if needed for Gemini format
                    # elif message["role"] == "tool":
                    #    # Assuming tool responses might have a different structure, adjust accordingly
                    #    if isinstance(message.get("content"), list): # Assuming tool content is a list of tool_code/tool_result
                    #        tool_parts = []
                    #        for part in message["content"]:
                    #            if isinstance(part, dict) and part.get("type") == "tool_code" and "text" in part:
                    #                 tool_parts.append({"functionCall": {"name": part.get("name", "unknown_tool"), "args": json.loads(part.get("text", "{}"))}}) # Assuming tool_code text is JSON args
                    #            elif isinstance(part, dict) and part.get("type") == "tool_result" and "text" in part:
                    #                 tool_parts.append({"functionResponse": {"name": part.get("name", "unknown_tool"), "response": {"content": part.get("text", "")}}}) # Assuming tool_result text is the response
                    #            else:
                    #                 logger.warning(f"Skipping tool message part due to unexpected format: {part}")
                    #        if tool_parts:
                    #            google_genai_conversation.append({
                    #                "role": "tool", # Or 'function' depending on Gemini version/format
                    #                "parts": tool_parts
                    #            })
                    #        else:
                    #            logger.warning(f"Skipping tool message with content list as no valid parts were found: {message}")
                    #    else:
                    #        logger.warning(f"Skipping tool message with unexpected content type or missing content: {message}")

                else:
                     logger.warning(f"Skipping invalid message format in payloads (during conversation build): {message}")


        logger.debug(f"google_genai_conversation: {google_genai_conversation}")

        # Added check for empty conversation before calling API
        if not google_genai_conversation:
             logger.warning("google_genai_conversation is empty, not calling generate_content.")
             # Return an empty or default LLMResponse, or raise an error
             return LLMResponse("assistant", completion_text="无法生成回复，对话内容为空或格式错误。")


        result = await self.client.generate_content(
            contents=google_genai_conversation,
            model=self.get_model(),
            system_instruction=system_instruction,
            tools=tool
        )
        logger.debug(f"result: {result}")

        # Added robust checking for the structure of the result
        if not isinstance(result, dict) or "candidates" not in result or not isinstance(result["candidates"], list) or not result["candidates"]:
            logger.error(f"Gemini API returned unexpected result format: {result}")
            # Check if the result contains an error message even if status was 200 (less likely but possible)
            if "error" in result and "message" in result["error"]:
                 raise Exception(f"Gemini API Returned Error: {result['error']['message']}")
            else:
                 raise Exception(f"Gemini API Returned Unexpected Response: {result}")

        first_candidate = result["candidates"][0]
        if not isinstance(first_candidate, dict) or "content" not in first_candidate or not isinstance(first_candidate["content"], dict) or "parts" not in first_candidate["content"] or not isinstance(first_candidate["content"]["parts"], list):
             logger.error(f"Gemini API returned unexpected candidate format: {first_candidate}")
             raise Exception(f"Gemini API Returned Unexpected Candidate Format: {first_candidate}")


        candidates_parts = first_candidate['content']['parts']
        llm_response = LLMResponse("assistant") # Default role
        llm_response.completion_text = "" # Initialize completion_text

        # Added robust checking for each part in candidates_parts
        for candidate_part in candidates_parts:
            if not isinstance(candidate_part, dict):
                 logger.warning(f"Skipping candidate part as it is not a dictionary: {candidate_part}")
                 continue

            if 'text' in candidate_part and isinstance(candidate_part['text'], str):
                llm_response.completion_text += candidate_part['text']
            elif 'functionCall' in candidate_part and isinstance(candidate_part['functionCall'], dict):
                # Assuming functionCall means the role should be 'tool'
                llm_response.role = "tool"
                # Added checks for name and args in functionCall
                func_call_name = candidate_part['functionCall'].get('name')
                func_call_args = candidate_part['functionCall'].get('args')

                if func_call_name is not None and func_call_args is not None:
                     llm_response.tools_call_name.append(func_call_name)
                     llm_response.tools_call_args.append(func_call_args) # args is already a dict
                else:
                     logger.warning(f"Skipping functionCall part due to missing name or args: {candidate_part['functionCall']}")

            # Add handling for functionResponse if needed to process previous tool calls
            # elif 'functionResponse' in candidate_part:
            #    # Process function response if Gemini returns them in the output structure
            #    pass # Implement based on expected functionResponse structure


        # If the role is still default 'assistant' but completion_text is empty,
        # and there were no tool calls, something might be wrong.
        if llm_response.role == "assistant" and not llm_response.completion_text and not llm_response.tools_call_name:
             logger.warning(f"Gemini API returned a candidate with no text and no function call: {first_candidate}")
             llm_response.completion_text = "（收到空回复，请稍后重试）" # Provide a fallback message


        return llm_response


    async def text_chat(
        self,
        prompt: str,
        session_id: str,
        image_urls: List[str]=None,
        func_tool: FuncCall=None,
        contexts=None,
        system_prompt=None,
        **kwargs
    ) -> LLMResponse:
        new_record = await self.assemble_context(prompt, image_urls)
        context_query = []
        # Ensure session_memory for session_id exists before accessing
        if session_id not in self.session_memory:
             self.session_memory[session_id] = [] # Initialize if not exists
             logger.info(f"Initialized session_memory for new session_id: {session_id}")


        if not contexts:
            # Ensure elements from session_memory are valid before adding
            valid_session_memory = [record for record in self.session_memory[session_id] if isinstance(record, dict) and 'role' in record and 'content' in record]
            context_query = [*valid_session_memory, new_record]
        else:
             # Ensure elements from contexts are valid before adding
             valid_contexts = [record for record in contexts if isinstance(record, dict) and 'role' in record and 'content' in record]
             context_query = [*valid_contexts, new_record]

        if system_prompt:
            # Ensure system_prompt is string before adding
            if isinstance(system_prompt, str):
                # Added checks to prevent adding if already present or handled differently
                # Ensure system prompt is added only once and at the beginning if needed by the LLM
                # The _query method extracts system_instruction separately, so adding here might be redundant or depend on desired behavior.
                # If the LLM specifically needs system prompt *in* the messages list, add it here.
                # Given how _query extracts system_instruction, adding here might lead to duplication in payload.
                # Let's rely on _query to handle the extracted system_instruction.
                # If you *must* include it in the messages list for some reason, ensure it's not already there.
                pass # Relying on _query to use the system_instruction parameter


        # Re-assemble payloads including system_prompt in the dedicated system_instruction field if present
        payloads = {
            "messages": context_query, # This list now only contains user/assistant/tool messages
            **self.provider_config.get("model_config", {})
        }
        # system_prompt is now passed as a separate argument to _query if needed
        # However, _query extracts system_instruction from the messages list.
        # This seems like a potential inconsistency in how system prompts are handled.
        # Let's adjust _query to accept system_prompt argument directly.
        # Or, ensure system_prompt is added to context_query *only* if the LLM requires it in messages AND the format is correct.
        # Given the original _query logic, it expects system message IN the messages list.
        # Let's revert to original _query structure expectation but keep validation.

        # Original logic: system prompt is added to context_query if present.
        # Let's ensure the system prompt structure is correct if added.
        if system_prompt and isinstance(system_prompt, str):
             system_message = {"role": "system", "content": system_prompt}
             # Check if a system message already exists in context_query to avoid duplicates
             if not any(isinstance(msg, dict) and msg.get('role') == 'system' for msg in context_query):
                 context_query.insert(0, system_message)
             else:
                 logger.warning("System prompt already exists in context_query, skipping adding new system_prompt.")
        elif system_prompt:
             logger.warning(f"Provided system_prompt is not a string, skipping: {system_prompt}")


        for part in context_query:
            # Added check for type before deleting key
            if isinstance(part, dict) and '_no_save' in part:
                del part['_no_save']

        payloads = {
            "messages": context_query,
            **self.provider_config.get("model_config", {})
        }


        try:
            # Pass system_prompt separately if _query was modified to accept it,
            # otherwise _query extracts it from messages as before.
            # Based on original _query, it extracts from messages, so no change here needed for now.
            llm_response = await self._query(payloads, func_tool)
            await self.save_history(contexts, new_record, session_id, llm_response)
            return llm_response
        # Added specific exception handling for timeout
        except (aiohttp.ClientTimeout, asyncio.TimeoutError) as e:
             logger.error(f"Gemini API request timed out: {e}")
             # You can return a specific error response or raise a custom exception here
             # For now, re-raising with a more user-friendly message
             raise Exception(f"请求 Gemini API 超时，请稍后重试。") from e
        except Exception as e:
            # Check if the exception message indicates context length error
            # Using a more robust check than just 'in str(e)' if possible,
            # but sticking to original logic pattern for now.
            if "context length" in str(e).lower(): # Case-insensitive check
                retry_cnt = 10
                while retry_cnt > 0:
                    logger.warning(f"请求失败：{e}。上下文长度超过限制。尝试弹出最早的记录然后重试。剩余重试次数：{retry_cnt}")
                    try:
                        # Ensure pop_record is successful and actually removes something relevant
                        popped_record = self.pop_record(session_id)
                        if popped_record:
                             logger.info(f"Popped record from session {session_id} for retry: {popped_record}")
                             # Re-assemble payloads with modified session memory
                             valid_session_memory = [record for record in self.session_memory[session_id] if isinstance(record, dict) and 'role' in record and 'content' in record]
                             retry_context_query = [*valid_session_memory, new_record]
                             if system_prompt and isinstance(system_prompt, str):
                                  system_message = {"role": "system", "content": system_prompt}
                                  if not any(isinstance(msg, dict) and msg.get('role') == 'system' for msg in retry_context_query):
                                      retry_context_query.insert(0, system_message)

                             retry_payloads = {
                                 "messages": retry_context_query,
                                 **self.provider_config.get("model_config", {})
                             }

                             llm_response = await self._query(retry_payloads, func_tool)
                             # If successful, save history for the successful retry state
                             await self.save_history(None, new_record, session_id, llm_response) # Save based on current session_memory
                             break # 重试成功则跳出循环
                        else:
                             logger.warning(f"pop_record did not remove an eligible record for session {session_id}. Cannot retry context length error.")
                             retry_cnt = 0 # Stop retrying if no record was popped
                             break # Exit retry loop

                    # Catch specific exceptions during retry as well
                    except (aiohttp.ClientTimeout, asyncio.TimeoutError) as retry_e:
                         logger.error(f"Retry attempt failed due to timeout: {retry_e}")
                         retry_cnt -= 1
                         # Continue the while loop for the next retry if retry_cnt > 0
                    except Exception as retry_e:
                        if "context length" in str(retry_e).lower(): # Check for context length error again
                            retry_cnt -= 1
                        else:
                           # If retry fails with a different error, re-raise it
                           raise retry_e

                if retry_cnt == 0 and "context length" in str(e).lower():
                    # If retries are exhausted and the original error was context length
                    raise Exception("多次尝试后上下文长度仍然超过限制，请尝试缩短对话或清理会话记录。") from e
                elif retry_cnt == 0:
                     # If retries exhausted but the last error was not context length, raise the last error
                     # This case is covered by the `else: raise retry_e` inside the loop, but adding here for clarity if needed.
                     pass # The last exception will be propagated


            else:
                # Handle other types of exceptions
                logger.error(f"请求 Gemini API 时发生未知错误: {e}")
                # Check if the error is from the API client after status check (less likely but possible)
                if "Gemini API Error:" in str(e):
                     raise e # Re-raise the specific API error
                # Check if the error indicates a problem with the model or request setup
                elif "invalid_argument" in str(e).lower() or "bad request" in str(e).lower():
                     raise Exception(f"请求 Gemini API 失败，请检查模型配置或输入格式：{e}") from e
                else:
                     raise Exception(f"请求 Gemini API 时发生未知错误：{e}") from e


    async def save_history(self, contexts: List, new_record: dict, session_id: str, llm_response: LLMResponse):
        if llm_response.role == "assistant" and session_id:
            # 文本回复
            if not contexts:
                # 添加用户 record - Added check before appending
                if isinstance(new_record, dict) and 'role' in new_record and 'content' in new_record:
                     self.session_memory[session_id].append(new_record)
                else:
                     logger.warning(f"Skipping saving invalid new_record to session_memory: {new_record}")

                # 添加 assistant record - Added check before appending
                if isinstance(llm_response.completion_text, str):
                    assistant_record = {
                        "role": "assistant",
                        "content": llm_response.completion_text
                    }
                    self.session_memory[session_id].append(assistant_record)
                else:
                     logger.warning(f"Skipping saving assistant response with non-string completion_text: {llm_response.completion_text}")

            else:
                # Filter and validate contexts before saving
                contexts_to_save = [item for item in contexts if isinstance(item, dict) and 'role' in item and 'content' in item and '_no_save' not in item]
                # Validate new_record and assistant_record before extending
                valid_records = [*contexts_to_save]
                if isinstance(new_record, dict) and 'role' in new_record and 'content' in new_record:
                     valid_records.append(new_record)
                else:
                     logger.warning(f"Skipping saving invalid new_record from contexts: {new_record}")

                if isinstance(llm_response.completion_text, str):
                    assistant_record = {
                        "role": "assistant",
                        "content": llm_response.completion_text
                    }
                    valid_records.append(assistant_record)
                else:
                     logger.warning(f"Skipping saving assistant response with non-string completion_text from contexts: {llm_response.completion_text}")

                self.session_memory[session_id] = valid_records


            # Added error handling for JSON dump and DB update
            try:
                db_history_json = json.dumps(self.session_memory[session_id], ensure_ascii=False)
                self.db_helper.update_llm_history(session_id, db_history_json, self.provider_config['type'])
            except TypeError as e:
                logger.error(f"Failed to dump session_memory to JSON for session {session_id}: {e}. Data: {self.session_memory[session_id]}")
                # Decide how to handle this - maybe clear session memory for this session?
            except Exception as e:
                 logger.error(f"Failed to update DB history for session {session_id}: {e}")


    async def forget(self, session_id: str) -> bool:
        # Added check if session_id exists before clearing
        if session_id in self.session_memory:
            self.session_memory[session_id] = []
            # Consider clearing from DB as well if save_history updates it
            try:
                 self.db_helper.update_llm_history(session_id, json.dumps([]), self.provider_config['type'])
                 logger.info(f"Cleared session memory and DB history for session_id: {session_id}")
                 return True
            except Exception as e:
                 logger.error(f"Failed to clear DB history for session {session_id}: {e}")
                 return False # Indicate failure if DB update fails
        else:
             logger.warning(f"Attempted to forget non-existent session_id: {session_id}")
             return False # Indicate session_id did not exist


    def get_current_key(self) -> str:
        # Added check if client exists
        if hasattr(self, 'client') and self.client:
             return self.client.api_key
        return None # Or raise an error if client is expected to always exist

    def get_keys(self) -> List[str]:
        return self.api_keys

    def set_key(self, key):
        # Added check if client exists
        if hasattr(self, 'client') and self.client:
             self.client.api_key = key
        else:
             logger.warning("Attempted to set key before client was initialized.") # Or raise an error


    async def assemble_context(self, text: str, image_urls: List[str] = None):
        '''
        组装上下文。
        '''
        # Ensure text is string
        if not isinstance(text, str):
             logger.warning(f"assemble_context received non-string text: {text}")
             text = str(text) # Convert to string for robustness


        if image_urls and isinstance(image_urls, list): # Ensure image_urls is a list
            user_content = {"role": "user","content": [{"type": "text", "text": text}]}
            for image_url in image_urls:
                if not isinstance(image_url, str): # Ensure each image_url is a string
                     logger.warning(f"Skipping invalid image_url (not a string): {image_url}")
                     continue

                try:
                    image_data = None
                    if image_url.startswith("http"):
                        # Added error handling for image download
                        image_path = await download_image_by_url(image_url)
                        if image_path:
                            image_data = await self.encode_image_bs64(image_path)
                            # Optional: Clean up downloaded image file
                            # import os
                            # os.remove(image_path)
                        else:
                             logger.warning(f"Failed to download image from URL: {image_url}")
                             continue # Skip this image if download fails
                    else:
                        image_data = await self.encode_image_bs64(image_url) # Assumes local path or base64://

                    if image_data: # Only add if encoding was successful
                        # Added basic check for image_data format if possible
                        user_content["content"].append({"type": "image_url", "image_url": {"url": image_data}})
                    else:
                        logger.warning(f"Failed to encode image data for URL/path: {image_url}")

                except Exception as e:
                     logger.error(f"Error processing image URL {image_url}: {e}")
                     logger.warning(f"Skipping image URL due to processing error: {image_url}")
                     continue # Skip image on error

            # Ensure content list is not empty if text part failed or was missing initially
            # Although text part is added unconditionally if text is string, validation is good.
            if not user_content["content"]:
                 logger.warning(f"assembled user_content content list is empty: {user_content}")
                 # Decide how to handle - maybe return None or raise error?
                 # Returning basic text content if no images were successfully processed
                 return {"role": "user","content": text}

            return user_content
        else:
             # Ensure text is string and return basic text content
             return {"role": "user","content": text}


    async def encode_image_bs64(self, image_url: str) -> str:
        '''
        将图片转换为 base64
        Assumes image_url is a local file path or a base64:// string.
        '''
        if not isinstance(image_url, str):
             logger.warning(f"encode_image_bs64 received non-string image_url: {image_url}")
             return '' # Return empty string for invalid input

        if image_url.startswith("base64://"):
            # Basic validation for base64 string format if needed
            return image_url.replace("base64://", "data:image/jpeg;base64,") # Assuming JPEG MIME type

        # Added error handling for file operations
        try:
            with open(image_url, "rb") as f:
                image_bs64 = base64.b64encode(f.read()).decode('utf-8')
                # Assuming JPEG MIME type, should be dynamic if possible
                return "data:image/jpeg;base64," + image_bs64
        except FileNotFoundError:
             logger.error(f"Image file not found for encoding: {image_url}")
             return ''
        except Exception as e:
             logger.error(f"Error encoding image file {image_url} to base64: {e}")
             return ''
