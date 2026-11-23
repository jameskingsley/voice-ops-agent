import json
import re
from pathlib import Path
from typing import Any


class VoiceOpsTools:
    """Operational tools for hands-free order, inventory, and policy checks."""

    def __init__(self, data_path: str = "data/ops_data.json") -> None:
        raw = json.loads(Path(data_path).read_text(encoding="utf-8-sig"))
        self.orders: dict[str, Any] = raw.get("orders", {})
        self.inventory: dict[str, Any] = raw.get("inventory", {})
        self.policies: list[dict[str, Any]] = raw.get("policies", [])

    def _normalize_order_id(self, raw_id: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_id.upper())
        if cleaned in self.orders:
            return cleaned
        # Extract digits (e.g., "ROD422" -> "422", "ORD4022" -> "4022")
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        if digits:
            exact_cand = f"ORD-{digits}"
            if exact_cand in self.orders:
                return exact_cand
            # Handle dropped zero in speech (e.g., "422" -> "4022", "421" -> "4021")
            if len(digits) == 3:
                padded_cand = f"ORD-{digits[0]}0{digits[1:]}"
                if padded_cand in self.orders:
                    return padded_cand
        return raw_id.strip().upper()

    def _normalize_sku(self, raw_sku: str) -> str:
        cleaned = raw_sku.strip().upper()
        if cleaned in self.inventory:
            return cleaned
        compact = re.sub(r"[^A-Z0-9]", "", cleaned)
        for sku_key in self.inventory:
            if re.sub(r"[^A-Z0-9]", "", sku_key) == compact:
                return sku_key
            # Match by trailing numeric code (e.g., "012" -> "SKU-PKG-012")
            digits = "".join(ch for ch in compact if ch.isdigit())
            if digits and sku_key.endswith(digits):
                return sku_key
        return cleaned

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "check_order_status",
                    "description": "Look up customer order payment status, bank transfer session reference, fulfillment status, and ETA.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "Order ID such as ORD-4021 or ORD-4022."}
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_sku_stock",
                    "description": "Look up available stock quantity, unit price in Naira, and reorder status for a SKU.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string", "description": "Product SKU such as SKU-PKG-012 or SKU-INV-500."}
                        },
                        "required": ["sku"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_policy",
                    "description": "Retrieve merchant operational rules for dispatch cutoffs, bank transfer reconciliation, or low stock thresholds.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Policy topic keyword such as dispatch, transfer, or reorder."}
                        },
                        "required": ["topic"],
                    },
                },
            },
        ]

    def execute(self, name: str, raw_args: str) -> str:
        args = json.loads(raw_args) if raw_args else {}
        if name == "check_order_status":
            oid = self._normalize_order_id(str(args.get("order_id", "")))
            record = self.orders.get(oid)
            if not record:
                return json.dumps({"found": False, "order_id": oid, "message": "Order ID not found."})
            return json.dumps({"found": True, **record})

        if name == "check_sku_stock":
            sku = self._normalize_sku(str(args.get("sku", "")))
            item = self.inventory.get(sku)
            if not item:
                return json.dumps({"found": False, "sku": sku, "message": "SKU not found."})
            needs_reorder = item["stock_qty"] <= item["reorder_point"]
            return json.dumps({"found": True, "needs_reorder": needs_reorder, **item})

        if name == "lookup_policy":
            topic = str(args.get("topic", "")).lower()
            matches = [
                p for p in self.policies
                if any(word in (p["title"] + " " + p["summary"]).lower() for word in topic.split())
            ] or self.policies
            return json.dumps({"policies": matches})

        return json.dumps({"error": f"Unknown tool: {name}"})
