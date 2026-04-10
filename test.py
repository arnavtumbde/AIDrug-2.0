# import subprocess
#
# def iupac_to_smiles(name):
#     result = subprocess.run(
#         ["java", "-jar", "opsin.jar"],
#         input=name + "\n",   # 🔥 IMPORTANT FIX
#         capture_output=True,
#         text=True
#     )
#
#     print("STDOUT:", result.stdout)
#     print("STDERR:", result.stderr)
#
#     return result.stdout.strip()
#
#
# print(iupac_to_smiles("ethanol"))

import requests
import urllib.parse


def get_iupac_from_cactus(smiles):
    # SMILES often contain characters that must be URL-encoded (like # or @)
    encoded_smiles = urllib.parse.quote(smiles)
    url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"

    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    return "Name not found"

print(get_iupac_from_cactus("CCO"))