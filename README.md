


# Python

**It is highly recommended to create a new isolated environment before continuing.**

## Install

`pip install -r requirements.txt`

## Run

`uvicorn app.main:app --host 0.0.0.0 --port 8000`

# Docker

## Build

`docker build . -t arcaff-api:0.1.0`

## Run

`docker run -p 8000:80 arcaff-api:0.1.0`

## Configuration

By default the model and data files will be store at `/arcnet/data` and `/arccnet/models`. These locations can be overridden by setting the `DATAPATH` and, or `MODESLPATH` environment variables. Additionally thes can be mount to host file system e.g. `-v /host/path:/arccnet/data`


# Test

## Cutout Classification

```
curl -X 'POST' \
  'http://127.0.0.1:8000/arcnet/classify_cutout/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "time": "2022-11-12T13:14:15+00:00",
  "hgs_latitude": 0,
  "hgs_longitude": 0
}'
```

```
{
  "time": "2022-11-12T13:14:15Z",
  "hgs_latitude": 0,
  "hgs_longitude": 0,
  "hale_class": "QS",
  "mcintosh_class": "QS"
}
```

## Full disk detection

```
curl -X 'POST' \
  'http://127.0.0.1:8000/arcnet/full_disk_detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "time": "2022-11-12T13:14:15+00:00"
}'
```


```
[
  {
    "time": "2022-11-12T13:14:15Z",
    "bbox": {
      "bottom_left": {
        "latitude": 20,
        "longitude": 18
      },
      "top_right": {
        "latitude": 30,
        "longitude": 28
      }
    },
    "hale_class": "Beta",
    "mcintosh_class": "Dao"
  },
  {
    "time": "2022-11-12T13:14:15Z",
    "bbox": {
      "bottom_left": {
        "latitude": 9,
        "longitude": 9
      },
      "top_right": {
        "latitude": 19,
        "longitude": 19
      }
    },
    "hale_class": "Beta-Gamma-Delta",
    "mcintosh_class": "Eki"
  },
  {
    "time": "2022-11-12T13:14:15Z",
    "bbox": {
      "bottom_left": {
        "latitude": 22,
        "longitude": 3
      },
      "top_right": {
        "latitude": 32,
        "longitude": 13
      }
    },
    "hale_class": "Alpha",
    "mcintosh_class": "Axx"
  }
]
```
