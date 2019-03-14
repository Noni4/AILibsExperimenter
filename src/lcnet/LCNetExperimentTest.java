package lcnet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.aeonbits.owner.ConfigCache;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.model.Component;
import hasco.model.ComponentInstance;
import hasco.model.ComponentUtil;
import hasco.model.Dependency;
import hasco.model.NumericParameterDomain;
import hasco.model.Parameter;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.PartialOrderedSet;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import weka.core.Instance;
import weka.core.Instances;

public class LCNetExperimentTest {

	public static void main(String[] args) {
		ILCNetConfig config = ConfigCache.getOrCreate(ILCNetConfig.class);
		
		ExperimentRunner experimentRunner = new ExperimentRunner(new IExperimentSetEvaluator() {
			
			@Override
			public IExperimentSetConfig getConfig() {
				return config;
			}
			
			@Override
			public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
					IExperimentIntermediateResultProcessor processor) throws Exception {
				Map<String, String> valuesOfKeyFields = experimentEntry.getExperiment().getValuesOfKeyFields();
				
				int seed = 1;
				Random random = new Random();
				
				double datasetSize = 0.5;
				
				String dataset = "iris";
				Instances data = new Instances(new BufferedReader(
						new FileReader(new File(config.getPath() + dataset + ".arff"))));
				data.setClassIndex(data.numAttributes() - 1);
				List<Instances> splits = WekaUtil.getStratifiedSplit(data, seed, datasetSize);
				Instances train = splits.get(0);
				Instances test = splits.get(1);
				
				ArrayList<String> providedInterfaces = new ArrayList<String>();
				ArrayList<Map<String, String>> requiredInterfaces = new ArrayList<Map<String, String>>();
				PartialOrderedSet<Parameter> parameters = new PartialOrderedSet<Parameter>();
				ArrayList<Dependency> dependecies = new ArrayList<Dependency>();
				
				NumericParameterDomain numericParameterDomain = new NumericParameterDomain(false, 1.0e-12, 10.0);
				Parameter parameter = new Parameter("R", numericParameterDomain, 1.0e-7);
				parameters.add(parameter);
				
				Component component = new Component("weka.classifiers.functions.Logistic", providedInterfaces, requiredInterfaces, parameters, dependecies);
				ComponentInstance componentInstance = ComponentUtil.randomParameterizationOfComponent(component, random);
				WEKAPipelineFactory wekaPipelineFactory = new WEKAPipelineFactory();
				MLPipeline mlPipeline = wekaPipelineFactory.getComponentInstantiation(componentInstance);
				mlPipeline.buildClassifier(train);
				
				double proportionOfCorretClassifiedInstances = 0.0;
				for (Instance instance : test) {
					if (mlPipeline.classifyInstance(instance) == instance.classValue()) {
						proportionOfCorretClassifiedInstances = proportionOfCorretClassifiedInstances + (1.0 / (double) test.size());
					}
				}
				
				Map<String, Object> results = new HashMap<>();
				results.put("accuracy", proportionOfCorretClassifiedInstances);
				results.put("datasetSize", datasetSize);
				results.put("seed", seed);
				results.put("Options", componentInstance.getParameterValues().toString());
				processor.processResults(results);
			}
		});

	}

}

