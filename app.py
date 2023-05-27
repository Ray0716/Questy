import gradio as gr

def contentText(conText):
    return conText

def combine(context, numQuestions):
    return 'Context:\n\n' + context + '\n\nExample:\n\nQ --> The sun is responsible for\nC --> { "text": [ "puppies learning new tricks", "children growing up and getting old", "flowers wilting in a vase", "plants sprouting, blooming and wilting" ], "label": [ "A", "B", "C", "D" ] }\nA --> D\n\nGenerate ' + str(numQuestions) + ' more multiple choice questions of this format whose answers can be found directly in the context provided.'

with gr.Blocks() as demo:
    content = gr.Textbox(label="Content", lines=2)
    nQuestions = gr.Slider(0, 25, label="Number of Questions", step=1)
    out = gr.Textbox(value="", label="Output")
    btn = gr.Button(value="Generate")
    btn.click(combine, inputs=[content, nQuestions], outputs=[out])
if __name__=="__main__":
    demo.launch()

