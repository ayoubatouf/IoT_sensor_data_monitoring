import json
from src.utils.file_io import load_from_json


def format_data_from_file(file_path):

    input_data = load_from_json(file_path)

    formatted_data = [
        {
            "footfall": footfall,
            "atemp": atemp,
            "selfLR": selfLR,
            "ClinLR": ClinLR,
            "DoleLR": DoleLR,
            "PID": PID,
            "outpressure": outpressure,
            "inpressure": inpressure,
            "temp": temp,
        }
        for footfall, atemp, selfLR, ClinLR, DoleLR, PID, outpressure, inpressure, temp in zip(
            input_data["footfall"],
            input_data["atemp"],
            input_data["selfLR"],
            input_data["ClinLR"],
            input_data["DoleLR"],
            input_data["PID"],
            input_data["outpressure"],
            input_data["inpressure"],
            input_data["temp"],
        )
    ]

    return {"data": formatted_data}


if __name__ == "__main__":
    data = format_data_from_file("inference_input.json")

    with open("request_input.json", "w") as output_file:
        json.dump(data, output_file, indent=4)
