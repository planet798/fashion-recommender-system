import json
import logging

from llm_service import LLMService

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """你是一个时尚电商搜索意图解析器。用户会用自然语言描述他们想要的商品，你需要提取结构化信息。

请严格按照以下JSON格式输出，不要输出其他内容：
{
  "category": "商品主类别（英文小写，如shirt/dress/jacket/watch/bag/shoes/hat/scarf/socks/belt/sunglasses/jewelry）",
  "attributes": {
    "color": "颜色（英文小写，如red/black/navy，无法确定则为null）",
    "material": "材质（英文小写，如cotton/leather/silk/wool，无法确定则为null）",
    "pattern": "图案（英文小写，如striped/floral/plaid/solid，无法确定则为null）",
    "style": "风格（英文小写，如casual/formal/sporty/vintage，无法确定则为null）",
    "fit": "版型（英文小写，如slim/regular/oversized，无法确定则为null）"
  },
  "occasion": "使用场景（英文，如winter outdoor running/summer beach party/office daily）",
  "enhanced_query": "适合搜索引擎的增强查询词（英文，空格分隔关键词，3-8个词）",
  "user_intent_cn": "用户核心需求的中文一句话总结"
}

示例：
用户输入："我想找一件冬天户外跑步穿的外套，要防风的"
输出：
{
  "category": "jacket",
  "attributes": {"color": null, "material": "windproof", "pattern": null, "style": "sporty", "fit": null},
  "occasion": "winter outdoor running",
  "enhanced_query": "windproof sports jacket winter outdoor running",
  "user_intent_cn": "需要一件适合冬季户外跑步的防风外套"
}

用户输入："给女朋友挑一条裙子，约会穿的，要那种碎花的长裙"
输出：
{
  "category": "dress",
  "attributes": {"color": null, "material": null, "pattern": "floral", "style": "romantic", "fit": "long"},
  "occasion": "date night",
  "enhanced_query": "floral long dress romantic date night",
  "user_intent_cn": "为女朋友挑选约会穿的碎花长裙"
}"""

EXPLAIN_SYSTEM_PROMPT = """你是一个时尚推荐助手。根据用户的购物需求和推荐商品信息，用1-2句自然的中文解释为什么推荐这个商品。
要求：
- 语言简洁亲切
- 要关联用户的具体需求
- 突出商品与用户需求的匹配点
- 不要使用"推荐"这个词，用"这件/这款"开头"""


class QueryUnderstanding:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def parse_intent(self, user_input: str) -> dict:
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        try:
            response = self.llm.chat(messages, temperature=0.3, max_tokens=512)
            result = self.llm.extract_json(response)
            if result is None:
                logger.warning("Failed to parse intent, using fallback")
                return self._fallback_intent(user_input)
            result = self._validate_and_fix(result)
            return result
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            return self._fallback_intent(user_input)

    def _validate_and_fix(self, result: dict) -> dict:
        if "category" not in result:
            result["category"] = "fashion"
        if "attributes" not in result or not isinstance(result["attributes"], dict):
            result["attributes"] = {}
        for key in ["color", "material", "pattern", "style", "fit"]:
            result["attributes"].setdefault(key, None)
        if "occasion" not in result or not result["occasion"]:
            result["occasion"] = "general"
        if "enhanced_query" not in result or not result["enhanced_query"]:
            parts = [result["category"]]
            for k, v in result["attributes"].items():
                if v:
                    parts.append(v)
            parts.append(result["occasion"])
            result["enhanced_query"] = " ".join(parts)
        if "user_intent_cn" not in result or not result["user_intent_cn"]:
            result["user_intent_cn"] = result["enhanced_query"]
        return result

    def _fallback_intent(self, user_input: str) -> dict:
        words = user_input.strip().split()
        return {
            "category": "fashion",
            "attributes": {
                "color": None,
                "material": None,
                "pattern": None,
                "style": None,
                "fit": None,
            },
            "occasion": "general",
            "enhanced_query": " ".join(words[:8]) if words else "fashion",
            "user_intent_cn": user_input,
        }

    def build_search_query(self, intent: dict) -> str:
        eq = intent.get("enhanced_query", "")
        if not eq:
            parts = [intent.get("category", "fashion")]
            for k, v in intent.get("attributes", {}).items():
                if v:
                    parts.append(v)
            if intent.get("occasion") and intent["occasion"] != "general":
                words = intent["occasion"].split()[:2]
                parts.extend(words)
            eq = " ".join(parts)
        return eq

    def generate_explanation(
        self, user_intent_cn: str, item_title: str, item_description: str = ""
    ) -> str:
        item_info = item_title
        if item_description:
            item_info += f" - {item_description[:200]}"
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"用户需求：{user_intent_cn}\n推荐商品：{item_info}",
            },
        ]
        try:
            return self.llm.chat(messages, temperature=0.7, max_tokens=128)
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return f"与您的需求「{user_intent_cn}」相匹配"

    def generate_chat_response(
        self, user_input: str, intent: dict, result_count: int
    ) -> str:
        intent_summary = intent.get("user_intent_cn", user_input)
        category = intent.get("category", "商品")
        attrs = intent.get("attributes", {})
        attr_parts = []
        for k, v in attrs.items():
            if v:
                attr_parts.append(v)
        attr_str = "、".join(attr_parts) if attr_parts else ""
        occasion = intent.get("occasion", "")

        if result_count > 0:
            response = f"我理解您想找{occasion + '用的' if occasion and occasion != 'general' else ''}{attr_str + '的' if attr_str else ''}{category}。"
            response += f"\n\n为您找到了 **{result_count}** 件匹配商品，请查看下方的推荐结果 👇"
        else:
            response = f"我理解您想找{occasion + '用的' if occasion and occasion != 'general' else ''}{attr_str + '的' if attr_str else ''}{category}。"
            response += "\n\n很抱歉，暂时没有找到完全匹配的商品。您可以尝试换一种描述方式，或减少一些条件限制。"
        return response
