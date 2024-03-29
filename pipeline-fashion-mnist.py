import kfp
from kfp import components
from kfp import dsl
from kfp import onprem

def data_prep_op(dataset_location, output):
  return dsl.ContainerOp(
    name='Data preparation',
    image='pascalschroeder/kf-dataprep',
    arguments=[
      '--dataset_location', dataset_location,
      '--output', output
    ],
    file_outputs={
      'output': '/prepdf_output.txt'
    }
  )

def feature_eng_op(prep_dataset_location, output):
  return dsl.ContainerOp(
    name='Feature Engineering',
    image='pascalschroeder/kf-featureeng',
    arguments=[
      '--dataset_location', prep_dataset_location,
      '--output', output
    ],
    file_outputs={
      'output': '/findf_output.txt'
    }
  )

def data_split_op(impr_dataset_location, test_size, random_state, output_train_img, output_train_label, output_test_img, output_test_label):
  return dsl.ContainerOp(
    name='Data Split',
    image='pascalschroeder/kf-datasplit',
    arguments=[
      '--dataset_location', impr_dataset_location,
      '--test_size', test_size,
      '--random_state', random_state,
      '--output_train_img', output_train_img,
      '--output_train_label', output_train_label,
      '--output_test_img', output_test_img,
      '--output_test_label', output_test_label
    ],
    file_outputs={
      'train_img': '/trainimg.txt',
      'train_label': '/trainlabel.txt',
      'test_img': '/testimg.txt',
      'test_label': '/testlabel.txt',
    }
  )

def model_build_op(input_shape_height, input_shape_width, num_units, num_outputs, activation_l2, activation_l3, optimizer, loss, metrics, model_output):
  return dsl.ContainerOp(
    name='Model building',
    image='pascalschroeder/kf-modelbuild',
    arguments=[
      '--input_shape_height', input_shape_height,
      '--input_shape_width', input_shape_width,
      '--num_units', num_units,
      '--num_outputs', num_outputs,
      '--activation_l2', activation_l2,
      '--activation_l3', activation_l3,
      '--optimizer', optimizer,
      '--loss', loss,
      '--metrics', metrics,
      '--output', model_output
    ],
    file_outputs={
      'output_model_loc': '/model.txt',
    }
  )

def model_download_op(input_shape_height, input_shape_width, model_output):
  return dsl.ContainerOp(
    name='Model download',
    image='pascalschroeder/kf-modeldownload',
    arguments=[
      '--input_shape_height', input_shape_height,
      '--input_shape_width', input_shape_width,
      '--output', model_output
    ],
    file_outputs={
      'output_model_loc': '/model.txt',
    }
  )

def model_train_op(input_train_img, input_train_label, input_shape_height, input_shape_width, model_location, epochs, model_output):
  return dsl.ContainerOp(
    name='Model training',
    image='pascalschroeder/kf-modeltrain',
    arguments=[
      '--input_train_img', input_train_img,
      '--input_train_label', input_train_label,
      '--input_shape_height', input_shape_height,
      '--input_shape_width', input_shape_width,
      '--model_location', model_location,
      '--epochs', epochs,
      '--output', model_output
    ],
    file_outputs={
      'output_model_loc': '/trained_model.txt',
    }
  )

def model_eval_op(input_test_img, input_test_label, input_shape_height, input_shape_width, model_location, result_output):
  return dsl.ContainerOp(
    name='Model evaluation',
    image='pascalschroeder/kf-modeleval',
    arguments=[
      '--input_test_img', input_test_img,
      '--input_test_label', input_test_label,
      '--input_shape_height', input_shape_height,
      '--input_shape_width', input_shape_width,
      '--model_location', model_location,
      '--output', result_output
    ],
    output_artifact_paths={
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json',
    },
  )

@dsl.pipeline(
  name='ML Test Pipeline',
  description='Test'
)

def pipeline(dataset_location='/mnt/data/manipulated_fashion_mnist.csv', test_size=0.3, random_state=42, input_shape_height=28, input_shape_width=28, use_pretrained_model='False', model_units_num=128, model_outputs_num=10, model_activation_func_layer2='relu', model_activation_func_layer3='softmax', optimizer='adam', loss='binary_crossentropy', metrics='accuracy', num_epochs=10, location_prepared_dataset='/mnt/data/prep_fashion_mnist.csv', location_improved_dataset='/mnt/data/impr_fasion_mnist.csv', location_training_images='/mnt/data/train_img.csv', location_training_labels='/mnt/data/train_labels.csv', location_test_images='/mnt/data/test_img.csv', location_test_labels='/mnt/data/test_labels.csv', location_base_model='/mnt/model/base_model.h5', location_trained_model='/mnt/model/trained_model.h5'):
  data_preparation = data_prep_op(dataset_location, location_prepared_dataset).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
  feature_engineering = feature_eng_op(data_preparation.outputs['output'], location_improved_dataset).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
  data_split = data_split_op(feature_engineering.outputs['output'], test_size, random_state, location_training_images, location_training_labels, location_test_images, location_test_labels).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
  
  with dsl.Condition(use_pretrained_model == 'True'):
    model_building = model_download_op(input_shape_height, input_shape_width, location_base_model).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
    model_training = model_train_op(data_split.outputs['train_img'], data_split.outputs['train_label'], input_shape_height, input_shape_width, model_building.outputs['output_model_loc'], num_epochs, location_trained_model).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
    model_evaluation = model_eval_op(data_split.outputs['test_img'], data_split.outputs['test_label'], input_shape_height, input_shape_width, model_training.outputs['output_model_loc'], '/mlpipeline-ui-metadata.json').apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))


  with dsl.Condition(use_pretrained_model == 'False'):
    model_building = model_build_op(input_shape_height, input_shape_width, model_units_num, model_outputs_num, model_activation_func_layer2, model_activation_func_layer3, optimizer, loss, metrics, location_base_model).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
    model_training = model_train_op(data_split.outputs['train_img'], data_split.outputs['train_label'], input_shape_height, input_shape_width, model_building.outputs['output_model_loc'], num_epochs, location_trained_model).apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))
    model_evaluation = model_eval_op(data_split.outputs['test_img'], data_split.outputs['test_label'], input_shape_height, input_shape_width, model_training.outputs['output_model_loc'], '/mlpipeline-ui-metadata.json').apply(onprem.mount_pvc("fashion-mnist-vol", 'local-storage', "/mnt"))



if __name__ == '__main__':
  kfp.compiler.Compiler().compile(pipeline, __file__ + '.tar.gz')