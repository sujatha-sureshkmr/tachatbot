/*
* (c) Copyright IBM Corporation 2018
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package com.ibm.mq.samples.jms;


import javax.jms.Destination;
import javax.jms.JMSConsumer;
import javax.jms.JMSContext;
import javax.jms.JMSException;
import javax.jms.JMSProducer;
import javax.jms.TextMessage;

import com.ibm.msg.client.jms.JmsConnectionFactory;
import com.ibm.msg.client.jms.JmsFactoryFactory;
import com.ibm.msg.client.wmq.WMQConstants;

/**
 * A minimal and simple application for Point-to-point messaging.
 *
 * Application makes use of fixed literals, any customisations will require
 * re-compilation of this source file. Application assumes that the named queue
 * is empty prior to a run.
 *
 * Notes:
 *
 * API type: JMS API (v2.0, simplified domain)
 *
 * Messaging domain: Point-to-point
 *
 * Provider type: IBM MQ
 *
 * Connection mode: Client connection
 *
 * JNDI in use: No
 *
 */
public class JmsPutGet {

	// System exit status value (assume unset value to be 1)
	private static int status = 1;

	// Create variables for the connection to MQ - // @cp4i-mq
	private static final String HOST = System.getenv("qmhostname"); // Host name or IP address
	private static final int PORT = 443; // Listener port for your queue manager
	private static final String CHANNEL = "FUNDTRANSFERCHL"; // Channel name
	private static final String QMGR = "fundtransfer"; // Queue manager name
	// private static final String APP_USER = "not_used"; // User name that application uses to connect to MQ
	// private static final String APP_PASSWORD = "not_used"; // Password that the application uses to connect to MQ
	// private static final String QUEUE_NAME = "TRANSFER.REQUEST"; // Queue that the application uses to put and get messages to and from
	private static final String FromCurr = System.getenv("FromCurr"); 
	private static final String ToCurr = System.getenv("ToCurr"); 
	private static final double CurrencyRate = Double.parseDouble(System.getenv("CurrencyRate")); 
	private static final double TransferAmount = Double.parseDouble(System.getenv("TransferAmount")); 
	private static final String QUEUE_NAME = System.getenv("QUEUE_NAME"); 

	/**
	 * Main method
	 *
	 * @param args
	 */
	public static void main(String[] args) {

		// Variables
		JMSContext context = null;
		Destination destination = null;
		JMSProducer producer = null;
		JMSConsumer consumer = null;



		try {
			// Create a connection factory
			JmsFactoryFactory ff = JmsFactoryFactory.getInstance(WMQConstants.WMQ_PROVIDER);
			JmsConnectionFactory cf = ff.createConnectionFactory();

			// Set the properties
			cf.setStringProperty(WMQConstants.WMQ_HOST_NAME, HOST);
			cf.setIntProperty(WMQConstants.WMQ_PORT, PORT);
			cf.setStringProperty(WMQConstants.WMQ_CHANNEL, CHANNEL);
			cf.setIntProperty(WMQConstants.WMQ_CONNECTION_MODE, WMQConstants.WMQ_CM_CLIENT);
			cf.setStringProperty(WMQConstants.WMQ_QUEUE_MANAGER, QMGR);
			cf.setStringProperty(WMQConstants.WMQ_APPLICATIONNAME, "JmsPutGet (JMS)");
			// cf.setBooleanProperty(WMQConstants.USER_AUTHENTICATION_MQCSP, true); // @cp4i-mq
			// cf.setStringProperty(WMQConstants.USERID, APP_USER); // @cp4i-mq
			// cf.setStringProperty(WMQConstants.PASSWORD, APP_PASSWORD); // @cp4i-mq
			cf.setStringProperty(WMQConstants.WMQ_SSL_CIPHER_SUITE, "*TLS12"); // @cp4i-mq
			
			// Create JMS objects
			context = cf.createContext();
			destination = context.createQueue("queue:///" + QUEUE_NAME);

			long uniqueNumber = System.currentTimeMillis() % 1000;
			
			double ToTransferAmount = TransferAmount * CurrencyRate;

			TextMessage message = context.createTextMessage("Your fundtransfer is in-progress of initialization. " + uniqueNumber);
			if (QUEUE_NAME.equals("TRANSFER.REQUEST")) 
			{
				message = context.createTextMessage("We are sending you a message to inform you that fundtransfer of "+FromCurr +" "+ TransferAmount + " ( " + ToCurr +" "+ ToTransferAmount +" ) to account number ####ABC that you have request is initiated. Refernece #" + uniqueNumber);
			}
			if (QUEUE_NAME.equals("TRANSFER.REPLY") )
			{
				message = context.createTextMessage("We are sending you a message to inform you that fundtransfer of "+FromCurr +" "+ TransferAmount + " ( " + ToCurr +" "+ ToTransferAmount +" ) to account number ####ABC that you have request is successfully completed. Refernece #" + uniqueNumber);
			}

			producer = context.createProducer();
			producer.send(destination, message);
			System.out.println("Sent message:\n" + message);

			consumer = context.createConsumer(destination); // autoclosable
			String receivedMessage = consumer.receiveBody(String.class, 15000); // in ms or 15 seconds

			System.out.println("\nReceived message:\n" + receivedMessage);

                        context.close();

			recordSuccess();
		} catch (JMSException jmsex) {
			recordFailure(jmsex);
		}

		System.exit(status);

	} // end main()

	/**
	 * Record this run as successful.
	 */
	private static void recordSuccess() {
		System.out.println("SUCCESS");
		status = 0;
		return;
	}

	/**
	 * Record this run as failure.
	 *
	 * @param ex
	 */
	private static void recordFailure(Exception ex) {
		if (ex != null) {
			if (ex instanceof JMSException) {
				processJMSException((JMSException) ex);
			} else {
				System.out.println(ex);
			}
		}
		System.out.println("FAILURE");
		status = -1;
		return;
	}

	/**
	 * Process a JMSException and any associated inner exceptions.
	 *
	 * @param jmsex
	 */
	private static void processJMSException(JMSException jmsex) {
		System.out.println(jmsex);
		Throwable innerException = jmsex.getLinkedException();
		if (innerException != null) {
			System.out.println("Inner exception(s):");
		}
		while (innerException != null) {
			System.out.println(innerException);
			innerException = innerException.getCause();
		}
		return;
	}

}
