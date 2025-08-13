import gradio as gr
import logging
from predict import ModelPredictor


class GradioApp:
    # Configuring logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='../logs/app.log', filemode='a')
    logger = logging.getLogger(__name__)

    def __init__(self, predictor):
        self.predictor = predictor
        self.demo = self.build_interface()
        self.logger.info("GradioApp initialized")

    def build_interface(self):
        with gr.Blocks(title="Skate Trick Learning Time Predictor") as demo:
            gr.Markdown("# Predict Time to Learn a New Skate Trick")
            gr.Markdown("Enter the skater's details to predict the time to land a new trick.")

            with gr.Row():
                age = gr.Number(label="Age")
                practice_hours = gr.Number(label="Practice Hours per Week", maximum=50)
                confidence = gr.Number(label="Confidence Level", maximum=10)
                motivation = gr.Number(label="Motivation Level", maximum=10)

            with gr.Row():
                gender = gr.Dropdown(label="Gender", choices=self.predictor.gender_options)
                experience_level = gr.Dropdown(label="Experience Level",choices=self.predictor.experience_level_options)

            with gr.Row():
                favorite_trick = gr.Dropdown(label="Favorite Trick", choices=self.predictor.favorite_trick_options)
                skateboard_type = gr.Dropdown(label="Skateboard Type", choices=self.predictor.skateboard_type_options)

            with gr.Row():
                learning_method = gr.Dropdown(label="Learning Method", choices=self.predictor.learning_method_options)
                previous_injuries = gr.Dropdown(label="Previous Injuries",
                                                choices=self.predictor.previous_injuries_options)
            self.logger.info("Input fields created")
            predict_button = gr.Button("Predict")
            clear_button = gr.Button("Clear")
            output = gr.Textbox(label="Prediction")

            predict_button.click(
                fn=self.predictor.predict,
                inputs=[age, practice_hours, confidence, motivation, gender, experience_level,
                        favorite_trick, skateboard_type, learning_method, previous_injuries],
                outputs=output
            )

            clear_button.click(
                fn=self.clear_inputs,
                inputs=[age, practice_hours, confidence, motivation, gender, experience_level,
                        favorite_trick, skateboard_type, learning_method, previous_injuries, output],
                outputs=[age, practice_hours, confidence, motivation, gender, experience_level,
                         favorite_trick, skateboard_type, learning_method, previous_injuries, output]
            )
        return demo

    def clear_inputs(self, age, practice_hours, confidence, motivation, gender, experience_level,
                     favorite_trick, skateboard_type, learning_method, previous_injuries, output):
        self.logger.info("Clearing input fields")
        return (20, 5.0, 5, 5,
                self.predictor.gender_options[0],
                self.predictor.experience_level_options[0],
                self.predictor.favorite_trick_options[0],
                self.predictor.skateboard_type_options[0],
                self.predictor.learning_method_options[0],
                self.predictor.previous_injuries_options[0],
                "")

    def launch(self):
        self.demo.launch(share=True)
        self.logger.info("Gradio interface launched")


if __name__ == "__main__":
    predictor = ModelPredictor()
    app = GradioApp(predictor)
    app.launch()
