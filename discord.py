from discordwebhook import Discord


class DiscordBot:
    def __init__(self):
        self.discord = Discord(url="https://discord.com/api/webhooks/1035170236665712660"
                                   "/ezpABEIu1K8g0BK8PJcpOjSHG0VC85SGJxylvR50iEmhM5iBsn3NyX12c3ykco01OtS7")

    def send_message(self, title=None, description=None, fields=[], file_names=[]):
        # fields = [
        #     {"name": "Epoch", "value": "4", "inline": True},
        #     {"name": "Accuracy", "value": "0.9976", "inline": True},
        #     {"name": "Loss", "value": "3.87", "inline": True}
        # ]
        # file_names = ["acc_plot_example.webp"]
        files = {filename: open(filename, "rb") for filename in file_names}
        self.discord.post(
            embeds=[{"title": title, "description": description, "fields": fields}],
            file=files
        )
