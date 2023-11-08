# API Documentation

## Product Photography Generation Endpoint

```
POST : /v1/txt2img     Image Generation API
```

### Payload
```python
payload = {
        "prompt": str,                                      # prompt for image, Dtype: string
        "negative_prompt": str,                             # negative prompt for image, Dtype: string
        "guidance_scale": int,                              # between 1 and 30, suggested value: 18, Dtype: int
        "num_inference_steps": int,                         # generation step, between 1 and 150, suggested value: 60, Dtype: int
        "width": int,                                       # result image width, between 64 and 2048, suggested value: 1024, Dtype: int
        "height": int,                                      # result image height, between 64 and 2048, suggested value: 1024, Dtype: int
    }
```
 So that's the backend. The API basically says what's available, what it's asking for, and where to send it. Now moving onto the frontend, I'll start with constructing a payload with the parameters I want. An example can be:
```python
{
    "prompt": "In a Surreal Lifestyle style, a vibrant yellow can with an explosion of lemons around it is settled into a bed of artistic tools.",
    "negative_prompt": "oversaturated, ugly, render, cartoon, grain, low-res, kitsch, anime, painting, bad, disfigured",                       
    "guidance_scale": 7,                        
    "num_inference_steps": 60,                          
    "width": 1024,                          
    "height": 1024,   
}
```
I can put in as few or as many parameters as I want in the payload. The API will use the defaults for anything I don't set.

After that, I can send it to the API.
 - Python
```python
    response = requests.post(url=f'{URL}/v1/txt2img', json=payload)
```
Again, this URL needs to match the web UI's URL. If we execute this code, the web UI will generate an image based on the payload. That's great, but then what? There is no image anywhere...


 - After the backend does its thing, the API sends the response back in a variable that was assigned above: response. The response contains three entries; images, parameters, and info, and I have to find some way to get the information from these entries.
 - First, I put this line r = response.json() to make it easier to work with the response.
 - "image" is a base64-encoded generated image.

### Responses
 #### 200: Successful Response
  - Media type : application/json
  - Example Value
 ```python
 {
    "image": "string"        # base64-encoded image
 } 
 ```

  #### 422: Validation Error
  - Media type : application/json
  - Example Value
 ```python
{
    "detail": [
        {
        "loc": [
            "string",
            0
        ],
        "msg": "string",
        "type": "string"
        }
    ]
}
 ```

### Example

#### Python
```python
    
    payload ={
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "width": 1024, 
                "height": 1024, 
                "num_images_per_prompt": 1, 
                "guidance_scale": 7,
                "num_inference_steps" : 60,
            }
    response = requests.post(url=f'{URL}/v1/txt2img', json=payload) 

    print(response.status_code)
    if response.status_code == 200:
        r = response.json()
        res_image =  r['image']
        image = Image.open(io.BytesIO(base64.b64decode(res_image.split(",",1)[0])))
```

