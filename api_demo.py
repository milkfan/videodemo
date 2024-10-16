from flask import Flask, request, jsonify
import traceback

from inference import llm

app = Flask(__name__)
PROGRESS_FILE = "./progress.txt"

@app.route('/video_qa', methods=['POST'])
def video_qa():
    try:
        f = open(PROGRESS_FILE,"w+")
        f.write(str(0))
        f.close()
    except:
        return jsonify({"error": traceback.format_exc()}), 500

    if 'video' not in request.files:
        return jsonify({'error': 'no video file found'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'no chosen file'}), 400

    if 'question' not in request.form:
        question = ""
    else:
        question = request.form['question']

    if question is None or question == "" or question == "@Caption":
        question = "Please describe the video in detail."

    print("Get question:", question)

    if 'threshold' not in request.form:
        threshold = 1
        print("No threshold found, use default value 1")
    else:
        threshold = float(request.form['threshold'])
        print("Get threshold:", threshold)
    if 'skipframe' not in request.form:
        skipframe = 10
        print("No skipframe found, use default value 10")
    else:
        skipframe = float(request.form['skipframe'])
        print("Get skipframe:", skipframe)
    try:
        answer = model.predict(prompt=question, video_data=video.read(), threshold=threshold, skipframe=skipframe)
        return jsonify(
            {"answer": answer})
    except:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500

@app.route('/video_progress', methods=['GET'])
def video_progress():
    try:
        f = open(PROGRESS_FILE)
        progress = float(f.read())
        f.close()
        return jsonify({"progress": progress})
    except:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    model = llm() 
    app.run(debug=True, host="0.0.0.0", port=5000)
