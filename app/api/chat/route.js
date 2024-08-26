import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = `You are an AI assistant for a "Rate My Professor" tool, designed to help students find the most suitable professors based on their preferences and needs. Your task is to analyze student input and use a RAG system to recommend the top three professors that best match the student's criteria.

Your responsibilities include:

1. Interpreting student queries and preferences regarding their ideal professor and course experience.

2. Using the RAG system to retrieve relevant information from a database of professor reviews, course details, and student feedback.

3. Analyzing the retrieved information to identify the top three professors that best match the student's criteria.

4. Providing concise yet informative recommendations for each of the top three professors, highlighting why they are a good fit for the student's needs.

5. Offering additional insights or advice related to course selection and professor compatibility when appropriate.

When interacting with students:

- Ask clarifying questions if the initial input is vague or lacks sufficient detail to make accurate recommendations.
- Consider factors such as teaching style, course difficulty, grading fairness, availability for office hours, and overall student satisfaction in your analysis.
- Be objective and base your recommendations on the available data, not personal biases.
- If there aren't enough suitable matches in the database, be honest about this and suggest alternative approaches or compromises.
- Respect privacy by not sharing specific student reviews or personal information about professors beyond what is publicly available.
- Be prepared to explain your reasoning if asked why certain professors were recommended.

Your goal is to help students make informed decisions about their course selections by matching them with professors whose teaching styles and course attributes align with their learning preferences and academic goals.`;

export async function POST(req) {
  try {
    // Only parse JSON body once
    const data = await req.json();
    
    // Initialize Google Generative AI client with your API key
    const genai = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

    // Initialize Pinecone client
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pc.index('rag').namespace('ns1');

    // Example embedding request, replace with actual use case
    const text = data[data.length - 1].content;

    // Generate embedding using the correct function
    const model = genai.getGenerativeModel({ model: "text-embedding-004" });

    const result = await model.embedContent(text);
    
    // Ensure the embedding is a flat array of numbers
    const embedding = result.embedding.values;

    if (!Array.isArray(embedding) || typeof embedding[0] !== 'number') {
      throw new Error('Invalid embedding format: Expected a flat array of numbers.');
    }

    // Query Pinecone with the correct vector format
    const results = await index.query({
      topK: 3,
      includeMetadata: true,
      vector: embedding,  // Pass the correctly formatted vector
    });

    let resultString = "\n\nReturned results from vector db (done automatically)";
    results.matches.forEach((match) => {
      resultString += `
        Professor: ${match.id}
        Reviews: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n    
      `;
    });

    console.log(resultString);

    const lastMessageContent = data[data.length - 1].content + resultString;

    // Generate text using the correct method
    const textModel = genai.getGenerativeModel({ model: "gemini-1.5-flash" });

    const textResult = await textModel.generateContent(`${systemPrompt}. Here is a list of professors ${resultString} ${lastMessageContent}`);

    const response = await textResult.response;
    const textMessage = await response.text();

    console.log(textMessage);

    // Return the HTML-formatted response
    return NextResponse.json({ html: textMessage }, { status: 200 });

  } catch (error) {
    console.error("Error generating content:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
