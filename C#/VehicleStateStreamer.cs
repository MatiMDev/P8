using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;
using Newtonsoft.Json;

public class VehicleStateStreamer : MonoBehaviour
{
    public Rigidbody rb;
    public string groundTruthURL = "http://127.0.0.1:5000/groundtruth";
    public string imuURL = "http://127.0.0.1:5000/imu";

    private Vector3 lastVelocity;
    private float lastTime;

    void Start()
    {
        if (rb == null)
            rb = GetComponent<Rigidbody>();

        lastVelocity = rb.linearVelocity;
        lastTime = Time.time;
        StartCoroutine(StreamData());
    }

    IEnumerator StreamData()
    {
        while (true)
        {
            SendGroundTruth();
            SendSimulatedIMU();
            yield return new WaitForSeconds(0.1f); // 10 Hz stream
        }
    }

    void SendGroundTruth()
    {
        var data = new Dictionary<string, object>
        {
            { "timestamp", Time.time },
            { "position", new float[]{ rb.position.x, rb.position.y, rb.position.z } },
            { "rotation", new float[]{ rb.rotation.eulerAngles.x, rb.rotation.eulerAngles.y, rb.rotation.eulerAngles.z } },
            { "velocity", new float[]{ rb.linearVelocity.x, rb.linearVelocity.y, rb.linearVelocity.z } },
            { "angular_velocity", new float[]{ rb.angularVelocity.x, rb.angularVelocity.y, rb.angularVelocity.z } }
        };

        StartCoroutine(PostData(groundTruthURL, data));
    }

    void SendSimulatedIMU()
    {
        float currentTime = Time.time;
        float dt = currentTime - lastTime;

        Vector3 acceleration = (rb.linearVelocity - lastVelocity) / Mathf.Max(dt, 0.0001f);
        lastVelocity = rb.linearVelocity;
        lastTime = currentTime;

        // Add random noise
        acceleration += Random.insideUnitSphere * 0.1f;
        Vector3 noisyGyro = rb.angularVelocity + Random.insideUnitSphere * 0.01f;

        var imu = new Dictionary<string, object>
        {
            { "timestamp", currentTime },
            { "acceleration", new float[]{ acceleration.x, acceleration.y, acceleration.z } },
            { "angularVelocity", new float[]{ noisyGyro.x, noisyGyro.y, noisyGyro.z } },
            { "orientation", new float[]{ rb.rotation.eulerAngles.x, rb.rotation.eulerAngles.y, rb.rotation.eulerAngles.z } }
        };


        StartCoroutine(PostData(imuURL, imu));
    }

    IEnumerator PostData(string url, Dictionary<string, object> payload)
    {
        string json = JsonConvert.SerializeObject(payload);
        var request = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogWarning("Post failed to " + url + ": " + request.error);
        }
    }
}
