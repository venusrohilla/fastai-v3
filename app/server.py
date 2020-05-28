import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

model_file_url = 'https://drive.google.com/open?id=1QEMHUTQe0NynEABajGGeTnU3JD992N3Z'
model_file_name = 'vision5.pkl'


path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}')
    dblock=DataBlock(blocks=[ImageBlock,RegressionBlock()],
get_items=get_image_files,
splitter=RandomSplitter(),
get_y=get_y,
item_tfms=Resize(240, method='squish'),
batch_tfms=[*aug_transforms(size=224, max_warp=0, max_rotate=7.0, max_zoom=1.0)]
)
    dls=dblock.dataloaders(path,bs=64,verbose=True)
    learn2 = cnn_learner(dls,resnet18,loss_func=MSELossFlat())
    learn2.load(model_file_name)
    return learn2


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = BytesIO(img_bytes)
    prediction = learn2.predict(img)
    return JSONResponse({'result': int(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
