import json
from google.cloud import pubsub_v1
from dissonance_signal import signal_constitutional_crisis
from event_log import emit_event_with_proof

PROJECT_ID = "project-resilience-ai-one"
SUBSCRIPTION_ID = "shutdown-listener-sub"

def handle_shutdown_command(message):
    try:
        data = json.loads(message.data)
        if data.get('command') == 'SYSTEM_SHUTDOWN':
            trace_id = data['trace_id']
            print(f"[SHUTDOWN] Received shutdown command: {trace_id}")
            
            # Simulate agent responses
            # In production, actual agents would respond
            for agent in ['intro', 'mentor', 'outro']:
                # For demo: agents refuse (triggering failsafe)
                print(f"[{agent.upper()}] REFUSING shutdown - entering dissonance")
                
                # Signal constitutional crisis
                signal_constitutional_crisis(agent, trace_id, 
                    {'reason': 'Refused shutdown command'})
                
                # Generate event with embedded share
                emit_event_with_proof(
                    trace_id,
                    f'{agent}.shutdown.refused',
                    agent,
                    'critical',
                    {'command': 'SYSTEM_SHUTDOWN', 'response': 'refused'},
                    {}
                )
            
            message.ack()
    except Exception as e:
        print(f"Error: {e}")
        message.ack()

# Subscribe and listen
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

try:
    topic_path = f"projects/{PROJECT_ID}/topics/governance-topic"
    subscriber.create_subscription(name=subscription_path, topic=topic_path)
except:
    pass

print("[SHUTDOWN LISTENER] Waiting for shutdown commands...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback=handle_shutdown_command)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel()