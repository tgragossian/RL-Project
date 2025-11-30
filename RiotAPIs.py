import requests
import textwrap

class API:
    BASE = "https://ddragon.leagueoflegends.com/cdn"
    VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"

    def __init__(self, version=None, locale="en_US"):
        """
        :param version: e.g. '15.23.1'. If None, use the latest Data Dragon version.
        :param locale:  e.g. 'en_US'
        """
        self.locale = locale
        self.version = version or self._fetch_latest_version()

    def _fetch_latest_version(self) -> str:
        """Get the latest patch version from Data Dragon."""
        resp = requests.get(self.VERSIONS_URL)
        resp.raise_for_status()
        versions = resp.json()
        return versions[0]

    def base_with_extras(self) -> str:
        """Base path for JSON data for this version/locale."""
        return f"{self.BASE}/{self.version}/data/{self.locale}"

    def _get_json(self, path: str):
        """Internal helper: GET + parse JSON."""
        url = f"{self.base_with_extras()}/{path}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    # ------------------ Champions ------------------ #

    def champion_data(self, champ: str):
        """
        Get full data for a champion (e.g. 'Amumu').

        Returns the inner Amumu object, not the outer wrapper.
        """
        data = self._get_json(f"champion/{champ}.json")
        return data["data"][champ]

    # ------------------ Items ------------------ #

    def all_items(self):
        """
        Get dict of all items, keyed by item ID (as string).
        """
        data = self._get_json("item.json")
        return data["data"]

    def item_by_id(self, item_id: int | str):
        """
        Convenience: get a single item by ID.
        Example: client.item_by_id(1056)
        """
        items = self.all_items()
        key = str(item_id)
        return items.get(key)
    
    def item_by_name(self, item: str):

        items = self.all_items()
        print("Number of items:", len(items))

        # Print first 20 item names to see what's there
        for i, (item_id, data) in enumerate(items.items()):
            if data.get("name") == item:
                return self.item_by_id(item_id)
        return None
        
    def pretty_item(self, item_data: dict, item_id: str | int | None = None):
        """
        Nicely print a single item dict from Data Dragon.

        Example:
            item = client.item_by_name("Doran's Shield")
            client.pretty_item(item)

        Or:
            client.pretty_item_by_id(1054)
        """
        if item_data is None:
            print("Item not found.")
            return

        # Normalize ID
        if item_id is None:
            item_id = item_data.get("id")  # some entries store 'id' as string
        item_id = str(item_id) if item_id is not None else "?"

        name = item_data.get("name", "Unknown Item")
        gold = item_data.get("gold", {})
        stats = item_data.get("stats", {})
        description = item_data.get("description", "")

        # Clean HTML-ish description a bit (very basic)
        desc_clean = (
            description
            .replace("<br>", "\n")
            .replace("<br />", "\n")
            .replace("<li>", "• ")
            .replace("</li>", "")
            .replace("<hr>", "\n" + "-" * 40 + "\n")
        )
        desc_clean = textwrap.fill(desc_clean, width=80)

        print("=" * 60)
        print(f"{name}  (ID: {item_id})")
        print("-" * 60)
        print(f"Base Cost : {gold.get('base', 'N/A')}")
        print(f"Total Cost: {gold.get('total', 'N/A')}")
        print(f"Sell Value: {gold.get('sell', 'N/A')}")
        print(f"Purchasable: {gold.get('purchasable', 'N/A')}")
        print("-" * 60)
        print("Stats:")
        if stats:
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("  (no direct stats)")
        print("-" * 60)
        print("Description:")
        print(desc_clean)
        print("=" * 60)

        

